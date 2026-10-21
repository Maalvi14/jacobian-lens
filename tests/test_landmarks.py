import pytest
import torch

from hlens.landmarks import compute_landmark_curves, save_landmarks, workspace_range
from jlens.examples import load_wikitext_prompts

N_PROMPTS = 30


@pytest.fixture(scope="module")
def curves(qwen):
    prompts = load_wikitext_prompts(N_PROMPTS)
    return compute_landmark_curves(
        qwen.model, qwen.lens, prompts, vocab_size=qwen.hf.config.get_text_config().vocab_size
    )


@pytest.mark.real_model
def test_landmark_curves_and_workspace_range(qwen, curves):
    c = curves
    print("\nlayer depth  top1   top5   top10  kurt(mean/med)  autocorr  effdim", flush=True)
    for i, l in enumerate(c.layers):
        print(
            f"  L{l:02d} {c.depth(l):.2f}  {c.topk_acc[1][i]:.3f}  {c.topk_acc[5][i]:.3f}  {c.topk_acc[10][i]:.3f}"
            f"  {c.kurtosis_mean[i]:8.1f}/{c.kurtosis_median[i]:6.1f}  {c.autocorr[i]:6.3f}  {c.eff_dim[i]:.3f}",
            flush=True,
        )
    wr = workspace_range(c)
    print(f"\nonset by metric: {wr.onset_by_metric}\noffset by metric: {wr.offset_by_metric}", flush=True)
    print(f"workspace range: layers {wr.onset}-{wr.offset} = depth {wr.onset_depth:.2f}-{wr.offset_depth:.2f}", flush=True)
    # CKA block structure: mean similarity of each layer to its neighbours vs to far layers
    cka = c.cka
    L = len(c.layers)
    adj = torch.stack([cka[i, i + 1] for i in range(L - 1)]).mean().item()
    far = torch.stack([cka[i, j] for i in range(L) for j in range(L) if abs(i - j) > 10]).mean().item()
    print(f"CKA adjacent-layer mean {adj:.3f}, >10 layers apart mean {far:.3f}", flush=True)
    print("CKA row for L12:", [f"{x:.2f}" for x in cka[12].tolist()], flush=True)

    save_landmarks(c, wr, "results/landmarks_qwen3.5-4b.json")

    # Qualitative signatures of paper Fig 28 / 27, thresholds set from the observed run with margin
    t1, ac = c.topk_acc[1], c.autocorr
    assert t1[-1] > 5 * t1[12], "top-1 lens accuracy should jump in the final layers"
    assert sum(ac[:3]) / 3 < 0.05, "autocorrelation should sit at/below the null early"
    assert max(ac) > 0.5, "persistent (autocorrelated) content in the middle band"
    assert ac[-1] < 0.5 * max(ac), "autocorrelation should fall back near the output"
    assert adj > far + 0.2, "CKA should show block structure"
    assert 0.2 < wr.onset_depth < 0.65 and 0.75 < wr.offset_depth <= 1.0
