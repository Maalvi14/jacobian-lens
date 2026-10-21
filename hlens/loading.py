from dataclasses import dataclass

import torch
import transformers

import jlens
from jlens.hf import HFLensModel

MODEL_ID = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_DIR = "qwen3.5-4b/jlens/Salesforce-wikitext"
LENS_FILE = f"{LENS_DIR}/Qwen3.5-4B_jacobian_lens.pt"
LENS_FILE_N1000 = f"{LENS_DIR}/Qwen3.5-4B_jacobian_lens_n1000.pt"


@dataclass(frozen=True)
class ModelSpec:
    """A model with a pre-fitted lens in `neuronpedia/jacobian-lens`."""

    key: str
    model_id: str
    lens_file: str
    fp32_gb: float  # weights alone in float32; curvature work needs this plus activations
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-4b": ModelSpec("qwen3.5-4b", MODEL_ID, LENS_FILE, 17.0),
    "qwen3.6-27b": ModelSpec(
        "qwen3.6-27b", "Qwen/Qwen3.6-27B",
        "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", 108.0,
        "64 layers (48 linear + 16 full attention), d=5120, untied lm_head. fp32 does not fit in 122 GB.",
    ),
    "gemma-3-12b": ModelSpec(
        "gemma-3-12b", "google/gemma-3-12b-pt",
        "gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt", 49.0,
        "gated on HF (accept the licence); the lens was fitted on the pretrained model",
    ),
    "llama3.1-8b": ModelSpec(
        "llama3.1-8b", "meta-llama/Llama-3.1-8B",
        "llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt", 32.0,
        "dense decoder (32 layers, d=4096, GQA, SwiGLU), base model; gated on HF (manual approval); the lens was fitted on 539 prompts of the base model",
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "google/gemma-3-12b-it",
        "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt", 49.0,
        "gated on HF (accept the licence)",
    ),
}


@dataclass(frozen=True)
class Loaded:
    hf: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizerBase
    model: HFLensModel
    lens: jlens.JacobianLens


def load_hf(
    *, dtype: torch.dtype = torch.bfloat16, device: str = "cuda", model_id: str = MODEL_ID
) -> torch.nn.Module:
    # eager attention: SDPA may dispatch to flash/mem-efficient kernels,
    # which don't support the double backward every HVP estimator needs
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    return hf.to(device).eval()


def load_qwen(
    *,
    dtype: torch.dtype = torch.bfloat16,
    lens_file: str = LENS_FILE,
    device: str = "cuda",
) -> Loaded:
    hf = load_hf(dtype=dtype, device=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tokenizer)  # compile=False by default
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=lens_file)
    return Loaded(hf=hf, tokenizer=tokenizer, model=model, lens=lens)


def load_model(
    key: str, *, dtype: torch.dtype = torch.bfloat16, device: str = "cuda", lens_file: str | None = None
) -> Loaded:
    """Load any model in `MODELS` with its pre-fitted lens (eager attention)."""
    spec = MODELS[key]
    hf = load_hf(dtype=dtype, device=device, model_id=spec.model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_id)
    model = jlens.from_hf(hf, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=lens_file or spec.lens_file)
    return Loaded(hf=hf, tokenizer=tokenizer, model=model, lens=lens)


def matched_layer(n_layers: int, ref_layer: int, *, ref_n_layers: int = 32) -> int:
    """The layer at the same depth fraction (layer / (n_layers - 1)) as `ref_layer` of a
    `ref_n_layers`-layer model (default: Qwen3.5-4B)."""
    return round(ref_layer / (ref_n_layers - 1) * (n_layers - 1))
