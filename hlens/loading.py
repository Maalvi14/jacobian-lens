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
class Loaded:
    hf: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizerBase
    model: HFLensModel
    lens: jlens.JacobianLens


def load_hf(*, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> torch.nn.Module:
    # eager attention: SDPA may dispatch to flash/mem-efficient kernels,
    # which don't support the double backward every HVP estimator needs
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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