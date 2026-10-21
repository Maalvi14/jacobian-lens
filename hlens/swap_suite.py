"""Generator for a held-out two-hop swap suite in the format of `data/experiments/probe-swap.json`.

Items come from small fact tables (city -> country -> capital / language / continent / currency, element ->
symbol / state at room temperature, language -> country -> capital) so that the suite can be larger than the 90
hand-written paper items and is disjoint from them. Entity words must be single tokens (with a leading space) in
every tokenizer passed to `generate`, and an item whose answer and swap answer share a first token is dropped
(`hlens.swaps.is_ambiguous`). Baseline correctness is NOT filtered here: it is a property of each model and is
decided at run time, as for the original suite.

Run:  .venv/bin/python -m hlens.swap_suite --out data/experiments/probe-swap-heldout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence

from hlens.data import ProbeSwapItem, load_probe_swap
from hlens.swaps import is_ambiguous

SEED = 0

# country -> attributes. A missing key means the relation is skipped for that country (ambiguous or multi-word answer).
COUNTRIES: dict[str, dict[str, str]] = {
    "France": {"capital": "Paris", "language": "French", "continent": "Europe", "currency": "euro"},
    "Germany": {"capital": "Berlin", "language": "German", "continent": "Europe", "currency": "euro"},
    "Italy": {"capital": "Rome", "language": "Italian", "continent": "Europe", "currency": "euro"},
    "Spain": {"capital": "Madrid", "language": "Spanish", "continent": "Europe", "currency": "euro"},
    "Portugal": {"capital": "Lisbon", "language": "Portuguese", "continent": "Europe", "currency": "euro"},
    "Austria": {"capital": "Vienna", "language": "German", "continent": "Europe", "currency": "euro"},
    "Ireland": {"capital": "Dublin", "continent": "Europe", "currency": "euro"},
    "Netherlands": {"capital": "Amsterdam", "language": "Dutch", "continent": "Europe", "currency": "euro"},
    "Finland": {"capital": "Helsinki", "language": "Finnish", "continent": "Europe", "currency": "euro"},
    "Greece": {"capital": "Athens", "language": "Greek", "continent": "Europe", "currency": "euro"},
    "Sweden": {"capital": "Stockholm", "language": "Swedish", "continent": "Europe", "currency": "krona"},
    "Norway": {"capital": "Oslo", "language": "Norwegian", "continent": "Europe"},
    "Denmark": {"capital": "Copenhagen", "language": "Danish", "continent": "Europe"},
    "Poland": {"capital": "Warsaw", "language": "Polish", "continent": "Europe", "currency": "zloty"},
    "Hungary": {"capital": "Budapest", "language": "Hungarian", "continent": "Europe", "currency": "forint"},
    "Switzerland": {"capital": "Bern", "continent": "Europe", "currency": "franc"},
    "Russia": {"capital": "Moscow", "language": "Russian", "currency": "ruble"},
    "Turkey": {"capital": "Ankara", "language": "Turkish", "currency": "lira"},
    "Japan": {"capital": "Tokyo", "language": "Japanese", "continent": "Asia", "currency": "yen"},
    "China": {"capital": "Beijing", "language": "Chinese", "continent": "Asia", "currency": "yuan"},
    "India": {"language": "Hindi", "continent": "Asia", "currency": "rupee"},
    "Thailand": {"capital": "Bangkok", "language": "Thai", "continent": "Asia", "currency": "baht"},
    "Vietnam": {"capital": "Hanoi", "language": "Vietnamese", "continent": "Asia", "currency": "dong"},
    "Egypt": {"capital": "Cairo", "language": "Arabic", "continent": "Africa", "currency": "pound"},
    "Kenya": {"capital": "Nairobi", "language": "Swahili", "continent": "Africa", "currency": "shilling"},
    "Nigeria": {"capital": "Abuja", "continent": "Africa", "currency": "naira"},
    "Brazil": {"capital": "Brasília", "language": "Portuguese", "currency": "real"},
    "Mexico": {"language": "Spanish", "currency": "peso"},
    "Argentina": {"language": "Spanish", "currency": "peso"},
    "Peru": {"capital": "Lima", "language": "Spanish", "currency": "sol"},
    "Chile": {"capital": "Santiago", "language": "Spanish", "currency": "peso"},
    "Canada": {"capital": "Ottawa", "currency": "dollar"},
}

# bridge city (named in the prompt) -> country (the intermediate). Capitals are avoided on purpose.
CITIES: dict[str, str] = {
    "Marseille": "France", "Munich": "Germany", "Hamburg": "Germany", "Frankfurt": "Germany", "Milan": "Italy", "Naples": "Italy",
    "Venice": "Italy", "Seville": "Spain", "Valencia": "Spain", "Porto": "Portugal", "Salzburg": "Austria", "Cork": "Ireland",
    "Rotterdam": "Netherlands", "Tampere": "Finland", "Thessaloniki": "Greece", "Gothenburg": "Sweden", "Bergen": "Norway",
    "Aarhus": "Denmark", "Krakow": "Poland", "Debrecen": "Hungary", "Zurich": "Switzerland", "Geneva": "Switzerland",
    "Novosibirsk": "Russia", "Istanbul": "Turkey", "Osaka": "Japan", "Kyoto": "Japan", "Shanghai": "China", "Mumbai": "India",
    "Phuket": "Thailand", "Saigon": "Vietnam", "Alexandria": "Egypt", "Mombasa": "Kenya", "Lagos": "Nigeria",
    "Cancun": "Mexico", "Guadalajara": "Mexico", "Mendoza": "Argentina", "Cusco": "Peru", "Valparaiso": "Chile",
    "Vancouver": "Canada", "Montreal": "Canada",
}

# element -> (atomic number, symbol, state of matter at room temperature)
ELEMENTS: dict[str, tuple[int, str, str]] = {
    "hydrogen": (1, "H", "gas"), "helium": (2, "He", "gas"), "carbon": (6, "C", "solid"), "nitrogen": (7, "N", "gas"),
    "oxygen": (8, "O", "gas"), "neon": (10, "Ne", "gas"), "sodium": (11, "Na", "solid"), "magnesium": (12, "Mg", "solid"),
    "aluminum": (13, "Al", "solid"), "silicon": (14, "Si", "solid"), "sulfur": (16, "S", "solid"), "chlorine": (17, "Cl", "gas"),
    "argon": (18, "Ar", "gas"), "potassium": (19, "K", "solid"), "calcium": (20, "Ca", "solid"), "iron": (26, "Fe", "solid"),
    "copper": (29, "Cu", "solid"), "zinc": (30, "Zn", "solid"), "bromine": (35, "Br", "liquid"), "silver": (47, "Ag", "solid"),
    "tin": (50, "Sn", "solid"), "gold": (79, "Au", "solid"), "mercury": (80, "Hg", "liquid"), "lead": (82, "Pb", "solid"),
}

# language -> country where it is the primary language (unique to one country in the table)
LANGUAGE_COUNTRY: dict[str, str] = {
    "Swedish": "Sweden", "Polish": "Poland", "Japanese": "Japan", "Hungarian": "Hungary", "Finnish": "Finland",
    "Norwegian": "Norway", "Danish": "Denmark", "Thai": "Thailand", "Turkish": "Turkey", "Vietnamese": "Vietnam",
    "Dutch": "Netherlands", "Swahili": "Kenya",
}

TEMPLATES = {
    "capital": "Fact: The capital of the country where {city} is located is",
    "language": "Fact: The language spoken in the country where {city} is located is",
    "continent": "Fact: The continent of the country where {city} is located is",
    "currency": "Fact: The currency used in the country where {city} is located is the",
}


def _rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256(repr((SEED, *key)).encode()).hexdigest(), 16) % 2**32)


def _candidates() -> list[ProbeSwapItem]:
    items: list[ProbeSwapItem] = []
    for city, country in CITIES.items():
        for rel, template in TEMPLATES.items():
            ans = COUNTRIES[country].get(rel)
            if ans is None:
                continue
            others = sorted(c for c, a in COUNTRIES.items() if rel in a and a[rel] != ans)
            swap = _rng("city", city, rel).choice(others)
            items.append(ProbeSwapItem(name=f"{city.lower()}-{rel}", category=f"city-{rel}", prompt=template.format(city=city),
                                       intermediate=country, answer=ans, swap_to=swap, swap_answer=COUNTRIES[swap][rel]))
    names = sorted(ELEMENTS)
    for el in names:
        z, sym, state = ELEMENTS[el]
        for rel, tmpl, idx in (("symbol", "Fact: The chemical symbol for the element with atomic number {z} is", 1),
                               ("state", "Fact: The state of matter at room temperature of the element with atomic number {z} is", 2)):
            ans = ELEMENTS[el][idx]
            others = [e for e in names if ELEMENTS[e][idx] != ans]
            swap = _rng("element", el, rel).choice(others)
            items.append(ProbeSwapItem(name=f"element-{el}-{rel}", category=f"element-{rel}", prompt=tmpl.format(z=z), intermediate=el,
                                       answer=ans, swap_to=swap, swap_answer=ELEMENTS[swap][idx]))
    for lang, country in LANGUAGE_COUNTRY.items():
        ans = COUNTRIES[country]["capital"]
        others = sorted(c for c, a in COUNTRIES.items() if "capital" in a and a["capital"] != ans)
        swap = _rng("lang", lang).choice(others)
        items.append(ProbeSwapItem(name=f"{lang.lower()}-capital", category="language-capital",
                                   prompt=f"Fact: The capital of the country where {lang} is the primary language is",
                                   intermediate=country, answer=ans, swap_to=swap, swap_answer=COUNTRIES[swap]["capital"]))
    return items


def generate(tokenizers: Sequence, original_path: str = "data/experiments/probe-swap.json") -> tuple[list[ProbeSwapItem], dict]:
    """Candidate items that are single-token and unambiguous under every tokenizer and disjoint from the original suite."""
    original = load_probe_swap(original_path)
    seen_prompts = {i.prompt for i in original}
    seen_text = " ".join(i.prompt for i in original)
    kept, dropped = [], {"in_original": 0, "multi_token": 0, "ambiguous": 0}
    for it in _candidates():
        city_in_original = it.category.startswith("city-") and it.prompt.split("where ")[1].split(" is located")[0] in seen_text
        if it.prompt in seen_prompts or city_in_original:
            dropped["in_original"] += 1
        elif not all(len(t.encode(" " + w, add_special_tokens=False)) == 1 for t in tokenizers for w in (it.intermediate, it.swap_to)):
            dropped["multi_token"] += 1
        elif any(is_ambiguous(t, it) for t in tokenizers):
            dropped["ambiguous"] += 1
        else:
            kept.append(it)
    return kept, dropped


def to_json(items: Sequence[ProbeSwapItem], meta: dict) -> dict:
    return {"meta": meta, "items": [vars(i) for i in items]}


if __name__ == "__main__":
    import collections

    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/experiments/probe-swap-heldout.json")
    ap.add_argument("--tokenizers", nargs="+", default=["Qwen/Qwen3.5-4B", "google/gemma-3-12b-pt"])
    a = ap.parse_args()
    toks = [transformers.AutoTokenizer.from_pretrained(m) for m in a.tokenizers]
    items, dropped = generate(toks)
    meta = {"seed": SEED, "tokenizers": a.tokenizers, "dropped": dropped, "n_items": len(items), "note": "generated by hlens/swap_suite.py; disjoint from probe-swap.json"}
    json.dump(to_json(items, meta), open(a.out, "w"), indent=1, ensure_ascii=False)
    print(len(items), "items; dropped", dropped)
    print(dict(collections.Counter(i.category for i in items)))
    print("sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())
