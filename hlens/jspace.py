import torch
import scipy.optimize

from dataclasses import dataclass

from jlens import JacobianLens, LensModel
from collections.abc import Sequence

def lens_vectors(
    lens: JacobianLens,
    model: LensModel,
    layer: int,
    token_ids: Sequence[int],
    *,
    h_ref: torch.Tensor | None = None,
    dim_batch: int = 8,
) -> torch.Tensor:
    if layer not in lens.source_layers:
        raise ValueError(f"layer={layer} not in lens.source_layers={lens.source_layers}")
    if h_ref is None:
        h_ref = torch.ones(model.d_model)
    
    n_tokens = len(token_ids)
    token_ids_tensor = torch.tensor(token_ids)
    u = torch.zeros(n_tokens, model.d_model, dtype=torch.float32)

    for start in range(0, n_tokens, dim_batch):
        end = min(start + dim_batch, n_tokens)
        n_this = end - start

        batch_h = h_ref.unsqueeze(0).expand(n_this, -1).clone().requires_grad_(True)
        logits = model.unembed(batch_h)

        cotangent = torch.zeros_like(logits)
        batch_indices = torch.arange(n_this)
        cotangent[batch_indices, token_ids_tensor[start:end]] = 1.0

        grads = torch.autograd.grad(outputs=logits, inputs=batch_h, grad_outputs=cotangent)[0]
        u[start:end] = grads.float()

    J_l = lens.jacobians[layer]
    return u @ J_l

    
def sparse_pursuit(
    h: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 25,
    tol: float = 1e-6,
) -> tuple[list[int], torch.Tensor]:
    norms = dictionary.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise ValueError(
            "dictionary atoms must be unit-norm (matching-pursuit correlation "
            "selection is scale-sensitive); max deviation from 1.0 was "
            f"{(norms - 1).abs().max().item():.4f}"
        )

    residual = h.clone()
    indices: list[int] = []
    coeffs = torch.zeros(0)

    for _ in range(k):
        correlations = dictionary @ residual
        correlations[indices] = -float("inf")

        best = int(correlations.argmax())
        if correlations[best] <= 0:
            break
        indices.append(best)

        selected = dictionary[indices]
        coeffs_np, _ = scipy.optimize.nnls(selected.numpy().T, h.numpy())
        coeffs = torch.from_numpy(coeffs_np).float()

        residual = h - selected.T @ coeffs
        if residual.norm() < tol:
            break

    return indices, coeffs


@dataclass(frozen=True)
class Decomposition:
    indices: list[int]
    coeffs: torch.Tensor
    jspace_component: torch.Tensor
    remainder: torch.Tensor
    frac_variance: float


def decompose(
    h: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 25,
) -> Decomposition:
    indices, coeffs = sparse_pursuit(h, dictionary, k=k)
    jspace_component = dictionary[indices].T @ coeffs
    remainder = h - jspace_component
    frac_variance = (jspace_component.norm() ** 2 / h.norm() ** 2).item()

    return Decomposition(
        indices=indices,
        coeffs=coeffs,
        jspace_component=jspace_component,
        remainder=remainder,
        frac_variance=frac_variance,
    )


def k_variation(
    h: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k_values: Sequence[int],
) -> dict[int, Decomposition]:
    return {k: decompose(h, dictionary, k=k) for k in k_values}