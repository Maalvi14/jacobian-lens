from hlens.data import load_lens_eval, load_probe_swap, load_ignition, load_capacity

def test_load_lens_eval_multihop():
    items = load_lens_eval("data/evaluations/lens-eval-multihop.json")

    assert len(items) > 0
    first = items[0]
    assert first.name == "carnival-ocean"
    assert first.prompt.startswith("Fact: The ocean")
    assert first.target == "Atlantic"
    assert first.intermediates == ["Brazil"]


def test_load_lens_eval_typo_has_no_target():
    items = load_lens_eval("data/evaluations/lens-eval-typo.json")

    assert all(item.target is None for item in items)


def test_load_probe_swap():
    items = load_probe_swap("data/experiments/probe-swap.json")

    assert len(items) > 0
    first = items[0]
    assert first.name == "amazon-language"
    assert first.category == "multihop"
    assert first.intermediate == "Brazil"
    assert first.answer == "Portuguese"
    assert first.swap_to == "Mexico"
    assert first.swap_answer == "Spanish"


def test_load_ignition():
    config = load_ignition("data/experiments/ignition.json")

    assert len(config.countries_12) == 12
    assert config.countries_12[0] == "France"

    assert config.idiom_pairs[0] == ("bread", "butter")
    assert config.scrambled_pairs[0] == ("bread", "chips")

    assert all(isinstance(p, tuple) and len(p) == 2 for p in config.idiom_pairs)
    assert "{W}" in config.ctx_templates[0]
    assert "{W}" in config.noun_ctx_templates[0]


def test_load_capacity():
    config = load_capacity("data/experiments/capacity.json")

    assert config.block_families == ["names", "countries", "surnames", "cities"]
    assert config.targets_per_family["names"] == 100

    assert len(config.candidate_pools) > 0
    names_pool = config.candidate_pools[0]
    assert names_pool.name == "names"
    assert names_pool.proto == ["names"]
    assert "Tom" in names_pool.pool