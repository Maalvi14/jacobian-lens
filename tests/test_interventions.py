import torch
from hlens.interventions import coordinate_swap

def test_coordinate_swap_then_swap_back_is_identity():
    torch.manual_seed(0)
    d_model = 8
    v_s = torch.randn(d_model)
    v_t = torch.randn(d_model)
    h = torch.randn(d_model)

    swapped = coordinate_swap(h, v_s, v_t, alpha=1.0)
    swapped_back = coordinate_swap(swapped, v_s, v_t, alpha=1.0)

    torch.testing.assert_close(swapped_back, h, rtol=1e-4, atol=1e-4)


def test_coordinate_swap_leaves_orthogonal_component_untouched():
    torch.manual_seed(1)
    d_model = 8
    v_s = torch.randn(d_model)
    v_t = torch.randn(d_model)

    # build an orthogonal complement via QR, then take an h that's a mix of
    # v_s/v_t plus a component along a direction outside their span
    V = torch.stack([v_s, v_t], dim=1)
    Q, _ = torch.linalg.qr(V, mode="complete")
    outside_direction = Q[:, 2]  # orthogonal to both v_s and v_t

    h = 2.0 * v_s + 3.0 * v_t + 5.0 * outside_direction

    swapped = coordinate_swap(h, v_s, v_t, alpha=1.0)

    outside_before = torch.dot(h, outside_direction)
    outside_after = torch.dot(swapped, outside_direction)
    torch.testing.assert_close(outside_after, outside_before, rtol=1e-4, atol=1e-4)