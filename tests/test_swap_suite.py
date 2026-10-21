import json

from hlens.data import load_probe_swap
from hlens.experiments.swap_heldout import summarize, wilson
from hlens.swap_suite import CITIES, COUNTRIES, ELEMENTS, LANGUAGE_COUNTRY, _candidates, generate, to_json


class _Tok:
    """Every entity word is one token; a word's first token is its first letter (so first-letter collisions count as ambiguous)."""

    def encode(self, s, add_special_tokens=False):
        s = s.strip()
        return [ord(s[0])] if s else []

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_tables_are_consistent():
    assert set(CITIES.values()) <= set(COUNTRIES)
    assert set(LANGUAGE_COUNTRY.values()) <= set(COUNTRIES)
    for c in LANGUAGE_COUNTRY.values():
        assert "capital" in COUNTRIES[c]
    assert len({v[0] for v in ELEMENTS.values()}) == len(ELEMENTS)  # atomic numbers are unique


def test_candidates_are_well_formed_and_deterministic():
    a, b = _candidates(), _candidates()
    assert [vars(i) for i in a] == [vars(i) for i in b]
    assert len({i.name for i in a}) == len(a) and len({i.prompt for i in a}) == len(a)
    for i in a:
        assert i.intermediate != i.swap_to and i.answer != i.swap_answer


def test_generate_is_disjoint_from_the_original_suite():
    items, dropped = generate([_Tok()])
    original = {i.prompt for i in load_probe_swap("data/experiments/probe-swap.json")}
    assert items and not ({i.prompt for i in items} & original)
    assert dropped["in_original"] > 0  # e.g. the iron / mercury atomic-number prompts and Barcelona-style city prompts
    json.dumps(to_json(items, {"x": 1}))  # serializable


def test_wilson_interval_and_summary_rules():
    lo, hi = wilson(30, 100)
    assert 0.2 < lo < 0.3 < hi < 0.4
    assert wilson(0, 0) == (0.0, 1.0)
    g2 = {"n_baseline_correct": 100, "by_alpha": {"1.25": {"hits": 30, "rate": 0.3, "random_control_hits": [2, 0, 3],
          "per_item": [{"category": "c", "hit": h} for h in [True] * 30 + [False] * 70]}}}
    s = summarize(g2)["1.25"]
    assert s["passed"] and s["by_category"] == {"c": "30/100"}
    g2["by_alpha"]["1.25"]["random_control_hits"] = [7, 0, 0]  # 30 < 5 * 7
    assert not summarize(g2)["1.25"]["passed"]


class _WordTok:
    """One token per whole word (case-sensitive); a leading space is part of the token."""

    def __init__(self):
        self.vocab = {}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab.setdefault(s, len(self.vocab))]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_fold_case_adds_lower_and_capitalised_forms_except_for_symbols():
    from hlens.data import ProbeSwapItem
    from hlens.swaps import first_token_ids, is_ambiguous, item_token_ids

    tok = _WordTok()
    plain = first_token_ids(tok, "euro")
    folded = first_token_ids(tok, "euro", fold_case=True)
    assert plain < folded and len(folded) == 4 + 0 or len(folded) >= 4  # euro, Euro with and without a leading space
    mk = lambda cat, a, b: ProbeSwapItem("n", cat, "p", "x", a, "y", b)
    ans, sw = item_token_ids(tok, mk("city-currency", "euro", "yen"), fold_case=True)
    assert tok.vocab["Euro"] in ans and tok.vocab[" Yen"] in sw
    ans_sym, _ = item_token_ids(tok, mk("element-symbol", "He", "Na"), fold_case=True)
    assert tok.vocab["He"] in ans_sym and tok.vocab.get("he") not in ans_sym  # case-sensitive for symbols
    # folding can create ambiguity that case-sensitive matching does not have
    assert is_ambiguous(tok, mk("city-currency", "Yen", "yen"), fold_case=True) and not is_ambiguous(tok, mk("city-currency", "Yen", "yen"))
