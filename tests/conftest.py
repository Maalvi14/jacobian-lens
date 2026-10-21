import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: needs Qwen3.5-4B on CUDA (run with -m real_model)"
    )


def pytest_collection_modifyitems(config, items):
    # Real-model tests only run when explicitly selected: a plain `pytest`
    # stays the fast TinyDecoder suite.
    if config.option.markexpr:
        return
    skip = pytest.mark.skip(reason="real-model test; run with -m real_model")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def qwen():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from hlens.loading import load_qwen

    return load_qwen()


@pytest.fixture(scope="session")
def qwen_fp32():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from hlens.loading import load_qwen

    return load_qwen(dtype=torch.float32)

def _release():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def gemma_fp32():
    """Gemma 3 12B in fp32 (~49 GB), module-scoped so it is freed before any other model loads."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from hlens.loading import load_model

    loaded = load_model("gemma-3-12b", dtype=torch.float32)
    yield loaded
    del loaded
    _release()


@pytest.fixture(scope="module")
def llama_fp32():
    """Llama 3.1 8B in fp32 (~32 GB), module-scoped so it is freed before any other model loads."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from hlens.loading import load_model

    loaded = load_model("llama3.1-8b", dtype=torch.float32)
    yield loaded
    del loaded
    _release()
