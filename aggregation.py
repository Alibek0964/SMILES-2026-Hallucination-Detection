"""
aggregation.py — v13

Changes from v12 (data-audit + literature-driven):

  1. NLI swap: mDeBERTa-v3-base-mnli-xnli → vectara/hallucination_evaluation_model
     (HHEM-2.1-Open).  HHEM was specifically trained on FEVER + VitaminC for
     factual-consistency / groundedness — exactly the A1 (contradiction) and
     A2 (ungrounded invention) cases that dominate this dataset (40 % + 40 %).
     mDeBERTa-base output 3-class probabilities; HHEM outputs a single
     consistency score in [0, 1].  We derive 3 features from it (raw score,
     1 - score, |score - 0.5| * 2 = confidence).

  2. Add mid-layer probing: SELECTED_LAYERS = (-1, -2, -4, -8, -13).  Layer
     -13 = layer 12 of 24 — the centre of semantic encoding for Qwen-class
     models (literature: "Linear Probe Accuracy Scales with Model Size").
     Hallucinations are a SEMANTIC phenomenon; the late layers (17-24)
     primarily encode "next-token plausibility", not "is this fact correct".

  3. Disk persistence for the consistency cache:
     /kaggle/working/_consistency_cache.pkl.  Saves the k=5 alternative-
     generation outputs across runs.  The first v13 extraction is still
     the full ~44 min (consistency cache empty); v14+ iterations on the
     same dataset reuse the cached generations and run in ~5-10 min.

Feature layout (use_geometric=True, all flags True):

    last+mean pool over 5 layers       :  2 * 5 * 896  =  8960
    geometric scalars (norms/drift/cos):              =    44
    HHEM consistency                   :              =     3
    self-consistency                   :              =     7
    lexical                            :              =     6
    -----------------------------------------------------------
    total                                              =  9020

Speed budget (T4):
    Identical to v12: HHEM is smaller than mDeBERTa-base (~184M flan-t5-base
    vs ~280M deberta-base) and has unlimited context, so no per-sample
    overhead change.  Generation still dominates at ~15 s/batch.

Robustness:
    HHEM uses trust_remote_code=True (Vectara ships custom prediction code).
    If the download or remote code fails, NLI features fall back to
    [0.5, 0.5, 0] (neutral, no confidence) for the rest of the run, with a
    single warning printed.  v12 fall-back path is unchanged.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import os
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Tokeniser truncation-side patch (v12) — must run BEFORE model.py loads
# ============================================================================

def _apply_qwen_truncation_patch() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    if getattr(AutoTokenizer.from_pretrained, "_smiles_v12_trunc_patched", False):
        return

    _orig = AutoTokenizer.from_pretrained

    def _patched(*args, **kwargs):
        tok = _orig(*args, **kwargs)
        name = args[0] if args else kwargs.get("pretrained_model_name_or_path", "")
        if "qwen" in str(name).lower():
            try:
                tok.truncation_side = "left"
            except Exception:
                pass
        return tok

    _patched._smiles_v12_trunc_patched = True
    AutoTokenizer.from_pretrained = _patched


_apply_qwen_truncation_patch()


# ============================================================================
# MPS speed patch — unchanged (eager+output_attentions → SDPA on Apple)
# ============================================================================

def _apply_mps_speed_patch() -> None:
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        return

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return

    if getattr(AutoModelForCausalLM.from_pretrained, "_smiles_v8_patched", False):
        return

    _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    def _wrap_model_forward(model: torch.nn.Module) -> None:
        _orig_forward = model.forward

        def _wrapped(*args, **kwargs):
            wanted_attn = bool(kwargs.pop("output_attentions", False))
            out = _orig_forward(*args, **kwargs)
            if wanted_attn and getattr(out, "attentions", None) is None:
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                hs = getattr(out, "hidden_states", None)
                if input_ids is not None and hs is not None:
                    B, T = input_ids.shape[:2]
                    ref = hs[-1]
                    dummy = torch.zeros(B, 1, T, T, dtype=ref.dtype, device=ref.device)
                    n_layers = max(len(hs) - 1, 1)
                    out.attentions = tuple(dummy for _ in range(n_layers))
            return out

        model.forward = _wrapped

    def _patched_from_pretrained(*args, **kwargs):
        if kwargs.get("attn_implementation") == "eager":
            kwargs["attn_implementation"] = "sdpa"
        model = _orig_from_pretrained(*args, **kwargs)
        try:
            _wrap_model_forward(model)
        except Exception:
            pass
        return model

    _patched_from_pretrained._smiles_v8_patched = True
    AutoModelForCausalLM.from_pretrained = _patched_from_pretrained


_apply_mps_speed_patch()


# ============================================================================
# Constants
# ============================================================================

QWEN_IM_START = 151644
QWEN_IM_END = 151645
QWEN_END_OF_TEXT = 151643
RESPONSE_OFFSET = 3   # FIXED in v12

QWEN_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
NLI_MODEL_NAME = "vectara/hallucination_evaluation_model"  # HHEM-2.1-Open (v13)
SENTENCE_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

N_NLI_FEATURES = 3   # HHEM gives 1 score; we expose [score, 1-score, confidence]

K_CONSISTENCY = 5
GEN_MAX_NEW_TOKENS = 80
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95
ENCODER_MAX_LENGTH = 128
SEM_CLUSTER_THRESHOLD = 0.85
N_CONSISTENCY_FEATURES = 7

N_LEXICAL_FEATURES = 6

SELECTED_LAYERS: tuple[int, ...] = (-1, -2, -4, -8, -13)  # +mid layer in v13

INCLUDE_NLI: bool = True
INCLUDE_CONSISTENCY: bool = True
INCLUDE_LEXICAL: bool = True


STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "not", "no", "so", "yet", "for", "of", "in", "on",
    "at", "to", "from", "by", "with", "as", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "i", "you", "he",
    "she", "we", "us", "our", "his", "her", "have", "has", "had", "do",
    "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "about", "into", "over", "than", "then",
    "there", "here", "what", "which", "who", "when", "where", "how",
    "why", "if", "because", "while", "also", "only", "any", "all",
    "some", "more", "most", "other", "such", "very", "just",
})

TEMPLATE_LEAK_RE = re.compile(
    r"(?i)("
    r"you are an ai|"
    r"you are a helpful assistant|"
    r"step[- ]by[- ]step|"
    r"<\|im_start\||<\|im_end\||<\|endoftext\||"
    r"as an ai|"
    r"step\s*1\s*[:\.]|"
    r"first,\s|"
    r"\n\n\n"
    r")"
)

WORD_RE = re.compile(r"[a-zA-Z]+")
PASSAGE_RE = re.compile(
    r"single brief but complete sentence\.\s*(.*?)\s*Here is the question:",
    re.DOTALL,
)


# ============================================================================
# Helpers
# ============================================================================

def _split_context_response(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()

    if input_ids is None:
        n_real = len(real_idx)
        split = int(n_real * 0.7)
        return real_idx[:split], real_idx[split:]

    im_start_positions = (input_ids == QWEN_IM_START).nonzero(as_tuple=True)[0]
    if len(im_start_positions) == 0:
        n_real = len(real_idx)
        split = int(n_real * 0.7)
        return real_idx[:split], real_idx[split:]

    response_start = int(im_start_positions[-1].item()) + RESPONSE_OFFSET
    context_idx = real_idx[real_idx < response_start]
    response_idx = real_idx[real_idx >= response_start]

    if len(response_idx) > 0:
        keep = input_ids[response_idx] != QWEN_END_OF_TEXT
        response_idx = response_idx[keep]
    if len(response_idx) == 0:
        response_idx = context_idx[-1:] if len(context_idx) > 0 else real_idx[-1:]

    return context_idx, response_idx


def _last_token(layer: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return layer.index_select(0, idx[-1:].to(layer.device)).squeeze(0)


def _mean_pool(layer: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if len(idx) == 0:
        return torch.zeros(layer.size(-1), device=layer.device, dtype=layer.dtype)
    return layer.index_select(0, idx.to(layer.device)).mean(dim=0)


def _cos_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)


# ============================================================================
# Hidden-state pooling + geometric features (now over 5 layers)
# ============================================================================

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    _, response_idx = _split_context_response(input_ids, attention_mask)
    response_idx = response_idx.to(hidden_states.device)

    last_tok_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        last_tok_parts.append(_last_token(layer, response_idx))
        mean_parts.append(_mean_pool(layer, response_idx))

    return torch.cat(last_tok_parts + mean_parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
    last_layer_attentions: torch.Tensor | None = None,
) -> torch.Tensor:
    _, response_idx = _split_context_response(input_ids, attention_mask)
    response_idx = response_idx.to(hidden_states.device)

    last_toks = [_last_token(hidden_states[i], response_idx) for i in SELECTED_LAYERS]
    means    = [_mean_pool(hidden_states[i], response_idx) for i in SELECTED_LAYERS]

    parts: list[torch.Tensor] = []

    for v in last_toks + means:
        parts.append(v.norm(p=2))

    for v in (last_toks[0], means[0]):
        parts.append(v.mean())
        parts.append(v.std(unbiased=False))
        parts.append(v.min())
        parts.append(v.max())

    for track in (last_toks, means):
        for j in range(len(track) - 1):
            parts.append(_cos_t(track[j], track[j + 1]))
            parts.append((track[j] - track[j + 1]).norm(p=2))

    parts.append(_cos_t(last_toks[0], last_toks[-1]))
    parts.append((last_toks[0] - last_toks[-1]).norm(p=2))
    parts.append(_cos_t(means[0], means[-1]))
    parts.append((means[0] - means[-1]).norm(p=2))

    parts.append(_cos_t(last_toks[0], means[0]))
    parts.append((last_toks[0] - means[0]).norm(p=2))

    emb_last = _last_token(hidden_states[0], response_idx)
    emb_mean = _mean_pool(hidden_states[0], response_idx)
    parts.append(_cos_t(emb_last, last_toks[0]))
    parts.append((emb_last - last_toks[0]).norm(p=2))
    parts.append(_cos_t(emb_mean, means[0]))
    parts.append((emb_mean - means[0]).norm(p=2))

    return torch.stack(parts, dim=0).to(hidden_states.dtype)


# ============================================================================
# Text decoding helpers (v12)
# ============================================================================

@functools.lru_cache(maxsize=1)
def _get_qwen_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_TOKENIZER_NAME)


def _decode_response(input_ids: torch.Tensor, resp_idx: torch.Tensor) -> str:
    if len(resp_idx) == 0:
        return ""
    tok = _get_qwen_tokenizer()
    return tok.decode(input_ids[resp_idx].cpu().tolist(), skip_special_tokens=True).strip()


def _decode_clean_context(
    input_ids: torch.Tensor, ctx_idx: torch.Tensor,
) -> tuple[str, str]:
    if len(ctx_idx) < 4:
        return "", ""
    tok = _get_qwen_tokenizer()
    clean_idx = ctx_idx[:-3]
    full = tok.decode(
        input_ids[clean_idx].cpu().tolist(), skip_special_tokens=True,
    ).strip()
    m = PASSAGE_RE.search(full)
    passage = m.group(1).strip() if m else full
    return full, passage


# ============================================================================
# NLI features — HHEM-2.1-Open (v13)
# ============================================================================

_nli_failed: bool = False
_nli_cache: dict[str, float] = {}


@functools.lru_cache(maxsize=1)
def _get_nli():
    """Returns (model, device).  Uses the custom HHEM predict() interface."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME,
        trust_remote_code=True,
    )
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device).eval()
    print(f"[v13][NLI] loaded HHEM-2.1-Open ({NLI_MODEL_NAME}) on {device}")
    return model, device


def _hhem_score(premise: str, hypothesis: str) -> float:
    """Single consistency score in [0, 1] (1 = supported, 0 = hallucinated).

    Falls back to 0.5 (neutral) on any failure and refuses to retry — a
    missing model never hangs the pipeline at every sample.
    """
    global _nli_failed
    if _nli_failed:
        return 0.5
    if not premise or not hypothesis:
        return 0.5

    key = hashlib.md5(
        (premise + "|||" + hypothesis).encode("utf-8")
    ).hexdigest()
    if key in _nli_cache:
        return _nli_cache[key]

    try:
        model, _dev = _get_nli()
    except Exception as e:
        print(f"[v13][NLI] failed to load {NLI_MODEL_NAME}: {e}")
        _nli_failed = True
        return 0.5

    try:
        with torch.no_grad():
            result = model.predict([(premise, hypothesis)])
    except Exception as e:
        print(f"[v13][NLI] predict() failed: {e}")
        _nli_failed = True
        return 0.5

    if hasattr(result, "flatten"):
        flat = result.flatten()
        score_t = flat[0] if flat.shape[0] > 0 else None
    elif isinstance(result, (list, tuple)) and result:
        score_t = result[0]
    else:
        score_t = result

    if score_t is None:
        return 0.5
    if hasattr(score_t, "cpu"):
        score = float(score_t.cpu().item() if hasattr(score_t, "item") else score_t.cpu())
    elif hasattr(score_t, "item"):
        score = float(score_t.item())
    else:
        score = float(score_t)

    score = max(0.0, min(1.0, score))
    _nli_cache[key] = score
    return score


def extract_nli_features(
    passage_text: str,
    response_text: str,
    *,
    target_dtype,
    target_device,
) -> torch.Tensor:
    """3 features derived from HHEM consistency score."""
    if not passage_text or not response_text:
        return torch.zeros(N_NLI_FEATURES, dtype=target_dtype, device=target_device)

    score = _hhem_score(passage_text, response_text)
    feats = torch.tensor(
        [
            score,                          # raw consistency
            1.0 - score,                    # hallucination probability
            abs(score - 0.5) * 2.0,         # confidence in [0, 1]
        ],
        dtype=target_dtype,
        device=target_device,
    )
    return feats


# ============================================================================
# Self-consistency / semantic-entropy features (with disk persistence in v13)
# ============================================================================

CONSISTENCY_CACHE_FILE = os.environ.get(
    "V13_CONSISTENCY_CACHE", "/kaggle/working/_consistency_cache.pkl",
)
_qwen_gen_failed: bool = False
_encoder_failed: bool = False
_consistency_cache: dict[str, np.ndarray] = {}
_consistency_dirty_since_save: int = 0


def _load_consistency_cache() -> None:
    global _consistency_cache
    if not os.path.exists(CONSISTENCY_CACHE_FILE):
        return
    try:
        with open(CONSISTENCY_CACHE_FILE, "rb") as f:
            loaded = pickle.load(f)
        valid = {
            k: v for k, v in loaded.items()
            if isinstance(v, np.ndarray) and v.shape == (N_CONSISTENCY_FEATURES,)
        }
        _consistency_cache = valid
        if valid:
            print(f"[v13] consistency cache: {len(valid)} entries loaded")
    except Exception as e:
        print(f"[v13] consistency cache load failed: {e}")


def _save_consistency_cache() -> None:
    if not _consistency_cache:
        return
    try:
        os.makedirs(os.path.dirname(CONSISTENCY_CACHE_FILE) or ".", exist_ok=True)
        tmp = CONSISTENCY_CACHE_FILE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_consistency_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CONSISTENCY_CACHE_FILE)
    except Exception as e:
        print(f"[v13] consistency cache save failed: {e}")


_load_consistency_cache()


@functools.lru_cache(maxsize=1)
def _get_qwen_for_generation():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_TOKENIZER_NAME,
        torch_dtype=torch.bfloat16,
    )
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device).eval()
    print(f"[v13][gen] loaded {QWEN_TOKENIZER_NAME} for generation on {device}")
    return tok, model, device


@functools.lru_cache(maxsize=1)
def _get_encoder():
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(SENTENCE_ENCODER_NAME)
    model = AutoModel.from_pretrained(SENTENCE_ENCODER_NAME)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device).eval()
    print(f"[v13][encoder] loaded {SENTENCE_ENCODER_NAME} on {device}")
    return tok, model, device


def _encode_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tok, model, dev = _get_encoder()
    inputs = tok(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=ENCODER_MAX_LENGTH,
    ).to(dev)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.detach().cpu().numpy().astype(np.float32)


def _generate_alternatives(
    prompt_input_ids: torch.Tensor,
    seed: int,
    k: int = K_CONSISTENCY,
) -> list[str]:
    tok, model, dev = _get_qwen_for_generation()

    if prompt_input_ids.dim() == 1:
        prompt_input_ids = prompt_input_ids.unsqueeze(0)
    prompt_input_ids = prompt_input_ids.to(dev)
    attention_mask = torch.ones_like(prompt_input_ids)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_input_ids, attention_mask=attention_mask,
            max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
            num_return_sequences=k,
            pad_token_id=tok.eos_token_id,
            eos_token_id=[QWEN_END_OF_TEXT, QWEN_IM_END],
        )

    prompt_len = prompt_input_ids.shape[1]
    new_tokens = out[:, prompt_len:]
    return [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]


def _semantic_entropy(sim_matrix: np.ndarray, threshold: float) -> float:
    n = sim_matrix.shape[0]
    if n == 0:
        return 0.0
    visited = np.zeros(n, dtype=bool)
    sizes: list[int] = []
    for i in range(n):
        if visited[i]:
            continue
        size = 1
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= threshold:
                visited[j] = True
                size += 1
        sizes.append(size)
    total = float(sum(sizes))
    if total == 0:
        return 0.0
    p = np.array(sizes, dtype=np.float64) / total
    return float(-(p * np.log(p + 1e-10)).sum())


def extract_consistency_features(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor,
    orig_response: str,
    ctx_idx: torch.Tensor,
    *,
    target_dtype,
    target_device,
) -> torch.Tensor:
    global _qwen_gen_failed, _encoder_failed, _consistency_dirty_since_save

    zeros = torch.zeros(N_CONSISTENCY_FEATURES, dtype=target_dtype, device=target_device)
    if input_ids is None or _qwen_gen_failed or _encoder_failed:
        return zeros
    if not orig_response or len(ctx_idx) == 0:
        return zeros

    cache_key = hashlib.md5(input_ids.detach().cpu().numpy().tobytes()).hexdigest()
    if cache_key in _consistency_cache:
        cached = _consistency_cache[cache_key]
        return torch.from_numpy(cached).to(dtype=target_dtype, device=target_device)

    prompt_ids = input_ids[ctx_idx]
    seed = int.from_bytes(
        hashlib.md5(prompt_ids.detach().cpu().numpy().tobytes()).digest()[:4],
        "big",
    ) & 0x7FFFFFFF

    try:
        alternatives = _generate_alternatives(prompt_ids, seed=seed, k=K_CONSISTENCY)
    except Exception as e:
        print(f"[v13][consistency] generation failed: {e}; disabling for run")
        _qwen_gen_failed = True
        return zeros

    alternatives = [a if a else "<empty>" for a in alternatives]

    try:
        embs = _encode_texts([orig_response] + alternatives)
    except Exception as e:
        print(f"[v13][consistency] encoding failed: {e}; disabling for run")
        _encoder_failed = True
        return zeros

    sim = embs @ embs.T
    n_alts = len(alternatives)

    orig_to_alts = sim[0, 1:]
    alt_to_alt = sim[1:, 1:]

    f_orig_alt_mean = float(orig_to_alts.mean()) if len(orig_to_alts) > 0 else 0.0
    f_orig_alt_min = float(orig_to_alts.min()) if len(orig_to_alts) > 0 else 0.0

    if n_alts > 1:
        triu = np.triu_indices(n_alts, k=1)
        alt_pairwise = alt_to_alt[triu]
        f_alt_pair_mean = float(alt_pairwise.mean())
        f_alt_pair_std = float(alt_pairwise.std())
    else:
        f_alt_pair_mean = 1.0
        f_alt_pair_std = 0.0

    sem_entropy = _semantic_entropy(alt_to_alt, threshold=SEM_CLUSTER_THRESHOLD)

    orig_len = len(orig_response)
    alt_lens = np.array([len(a) for a in alternatives], dtype=np.float64)
    mean_alt_len = float(alt_lens.mean()) if len(alt_lens) > 0 else 1.0
    std_alt_len = float(alt_lens.std()) if len(alt_lens) > 0 else 0.0
    f_len_ratio = orig_len / max(mean_alt_len, 1.0)
    f_len_cov = std_alt_len / max(mean_alt_len, 1.0)

    feats_np = np.array(
        [f_orig_alt_mean, f_orig_alt_min, f_alt_pair_mean, f_alt_pair_std,
         sem_entropy, f_len_ratio, f_len_cov],
        dtype=np.float32,
    )
    _consistency_cache[cache_key] = feats_np
    _consistency_dirty_since_save += 1
    if _consistency_dirty_since_save >= 50:
        _save_consistency_cache()
        _consistency_dirty_since_save = 0
    return torch.from_numpy(feats_np).to(dtype=target_dtype, device=target_device)


# ============================================================================
# Lexical features (v12)
# ============================================================================

def _content_words(text: str) -> set[str]:
    return {
        w for w in (m.group(0).lower() for m in WORD_RE.finditer(text))
        if len(w) >= 3 and w not in STOP_WORDS
    }


def _token_recall_content(response: str, passage: str) -> float:
    resp = _content_words(response)
    if not resp:
        return 1.0
    ctx = _content_words(passage)
    return len(resp & ctx) / len(resp)


def _has_template_leak(response: str) -> float:
    return 1.0 if TEMPLATE_LEAK_RE.search(response) else 0.0


def _ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    n_ascii = sum(1 for c in text if ord(c) < 128)
    return n_ascii / len(text)


def _repeated_5grams_count(response: str) -> float:
    words = response.split()
    if len(words) < 5:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for i in range(len(words) - 4):
        gram = tuple(words[i:i + 5])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return float(repeats)


def extract_lexical_features(
    response_text: str,
    passage_text: str,
    *,
    target_dtype,
    target_device,
) -> torch.Tensor:
    if not response_text:
        feats = np.zeros(N_LEXICAL_FEATURES, dtype=np.float32)
    else:
        recall = _token_recall_content(response_text, passage_text)
        log_chars = float(np.log1p(len(response_text)))
        log_words = float(np.log1p(len(response_text.split())))
        leak = _has_template_leak(response_text)
        ar = _ascii_ratio(response_text)
        repeats = _repeated_5grams_count(response_text)
        feats = np.array(
            [recall, log_chars, log_words, leak, ar, repeats],
            dtype=np.float32,
        )
    return torch.from_numpy(feats).to(dtype=target_dtype, device=target_device)


# ============================================================================
# Disk feature cache (v13: dim 9020)
# ============================================================================

FEATURE_CACHE_FILE = os.environ.get(
    "V13_FEATURE_CACHE", "/kaggle/working/_v13_features.pkl",
)
EXPECTED_FEATURE_DIM = 9020   # 8960 + 44 + 3 + 7 + 6
_feature_cache: dict[str, np.ndarray] = {}
_feature_dirty_since_save = 0


def _load_feature_cache() -> None:
    global _feature_cache
    if not os.path.exists(FEATURE_CACHE_FILE):
        return
    try:
        with open(FEATURE_CACHE_FILE, "rb") as f:
            loaded = pickle.load(f)
        valid = {
            k: v for k, v in loaded.items()
            if isinstance(v, np.ndarray) and v.shape == (EXPECTED_FEATURE_DIM,)
        }
        _feature_cache = valid
        if len(valid) < len(loaded):
            print(
                f"[v13] feature cache: {len(valid)} valid, "
                f"{len(loaded) - len(valid)} stale entries discarded"
            )
        else:
            print(f"[v13] feature cache: {len(valid)} entries loaded")
    except Exception as e:
        print(f"[v13] feature cache load failed: {e}")


def _save_feature_cache() -> None:
    if not _feature_cache:
        return
    try:
        os.makedirs(os.path.dirname(FEATURE_CACHE_FILE) or ".", exist_ok=True)
        tmp = FEATURE_CACHE_FILE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_feature_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, FEATURE_CACHE_FILE)
    except Exception as e:
        print(f"[v13] feature cache save failed: {e}")


_load_feature_cache()


def _save_all_caches() -> None:
    _save_feature_cache()
    _save_consistency_cache()


atexit.register(_save_all_caches)


# ============================================================================
# Entry point — called once per sample by solution.py
# ============================================================================

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    *,
    input_ids: torch.Tensor | None = None,
    last_layer_attentions: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    global _feature_dirty_since_save

    pooled_dtype = torch.float32
    pooled_device = hidden_states.device

    cache_key = None
    if input_ids is not None and use_geometric:
        cache_key = hashlib.md5(
            input_ids.detach().cpu().numpy().tobytes()
            + attention_mask.detach().cpu().numpy().tobytes()
        ).hexdigest()
        if cache_key in _feature_cache:
            return torch.from_numpy(_feature_cache[cache_key]).to(
                dtype=pooled_dtype, device=pooled_device,
            )

    pooled = aggregate(hidden_states, attention_mask, input_ids=input_ids)
    if not use_geometric:
        return pooled

    geom = extract_geometric_features(
        hidden_states, attention_mask,
        input_ids=input_ids, last_layer_attentions=last_layer_attentions,
    )

    parts = [pooled, geom]

    ctx_idx, resp_idx = _split_context_response(input_ids, attention_mask)
    response_text = _decode_response(input_ids, resp_idx) if input_ids is not None else ""
    _, passage_text = (
        _decode_clean_context(input_ids, ctx_idx) if input_ids is not None else ("", "")
    )

    if INCLUDE_NLI:
        parts.append(extract_nli_features(
            passage_text, response_text,
            target_dtype=pooled.dtype, target_device=pooled.device,
        ))

    if INCLUDE_CONSISTENCY:
        parts.append(extract_consistency_features(
            input_ids, attention_mask,
            orig_response=response_text, ctx_idx=ctx_idx,
            target_dtype=pooled.dtype, target_device=pooled.device,
        ))

    if INCLUDE_LEXICAL:
        parts.append(extract_lexical_features(
            response_text, passage_text,
            target_dtype=pooled.dtype, target_device=pooled.device,
        ))

    feat = torch.cat(parts, dim=0)

    if cache_key is not None and feat.shape[0] == EXPECTED_FEATURE_DIM:
        _feature_cache[cache_key] = feat.detach().cpu().numpy()
        _feature_dirty_since_save += 1
        if _feature_dirty_since_save >= 50:
            _save_feature_cache()
            _feature_dirty_since_save = 0

    return feat
