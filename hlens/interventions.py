import torch
from torch import nn
from collections.abc import Callable, Sequence, Iterable

class ActivationEditor:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        edit_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._edit_fn = edit_fn
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = self._edit_fn(tensor)
        if torch.is_tensor(output):
            return edited
        return (edited, *output[1:])

    def __enter__(self) -> "ActivationEditor":
        try:
            for index in self._indices:
                self._handles.append(self._blocks[index].register_forward_hook(self._hook))
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    
def steer(h: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    return h + alpha * v


def ablate(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_hat = v / v.norm()
    projection = (h * v_hat).sum(dim=-1, keepdim=True) * v_hat
    return h - projection


def coordinate_swap(
    h: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    V = torch.stack([v_s, v_t], dim=1)
    V_pinv = torch.linalg.pinv(V)

    c = h @ V_pinv.T          
    c_swapped = c.flip(-1)    

    delta = alpha * (c_swapped - c) 
    return h + delta @ V.T