"""H10 at scale (100 pairs per family), per hlens/docs/preregistration_h10_scaled.md.

Families (pair = the unit): persona (role/character nouns), near-synonym / register-variant nouns, matched object-noun controls, and (reference) all country pairs.
Run:  .venv/bin/python -m hlens.experiments.h10_scaled pairs     # generate pairs, measure endpoint distances, match controls (no curvature)
      .venv/bin/python -m hlens.experiments.h10_scaled run       # trials (checkpointed, resumable)
      .venv/bin/python -m hlens.experiments.h10_scaled analyze
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys

import numpy as np
from scipy import stats

from hlens.data import load_ignition
from hlens.experiments.h10_control import endpoint_distance, match, match_quality, pair_dbar
from hlens.experiments.register_persona import family_signature, run

PREREG = "hlens/docs/preregistration_h10_scaled.md"
PAIRS = "results/h10_scaled_pairs.json"
OUT = "results/h10_scaled_qwen3.5-4b.json"
CKPT = "results/h10_scaled_qwen3.5-4b.partial.jsonl"
CARRIERS, N = slice(12, 14), 100
SEED = 0

PERSONA_POOL = """doctor pirate teacher soldier king beggar wizard farmer nurse pilot poet judge clown monk sailor queen witch chef knight scientist ghost princess vampire
mayor dragon banker hunter surgeon outlaw professor giant priest bandit butler cowboy explorer jester thief detective spy robot alien angel devil fairy elf goblin warrior assassin
merchant carpenter artist singer dancer student president general captain prisoner miner sheriff emperor prophet ninja samurai viking gladiator mechanic baker lawyer actor painter
shepherd sorcerer zombie mummy""".split()
OBJECT_POOL = """anchor pencil river sandwich mountain violin candle hammer ladder pizza window banana bottle saddle blanket trumpet lantern carrot umbrella helmet bridge cookie pillow wrench
basket rocket mirror pumpkin kettle compass garden telescope button volcano spoon castle feather engine clock melon shovel piano rope lemon bucket guitar needle tractor sofa cherry tent
microscope stamp trousers magnet onion drum sock truck lamp tower biscuit anvil carpet ball brush chair door fork glove hat key knife lock map mug nail paper plate ring scissors shoe soap
stone table towel wheel wire bell bowl box cup flag jar net pan pipe pot rug sail sponge string tray vase whistle barrel chain coin curtain hook ruler bicycle bread cheese orange apple
tomato potato""".split()
# near-synonym / register-variant noun pairs (formal-informal, British-American, plain synonyms); the first N single-token pairs are used
SYNONYM_PAIRS = [("lavatory", "toilet"), ("spouse", "wife"), ("adolescent", "teen"), ("individual", "person"), ("inhabitant", "resident"), ("garment", "clothing"), ("domicile", "home"),
    ("dwelling", "house"), ("vehicle", "car"), ("equine", "horse"), ("bovine", "cow"), ("toddler", "kid"), ("juvenile", "child"), ("offspring", "child"), ("sibling", "brother"), ("mentor", "coach"),
    ("companion", "friend"), ("acquaintance", "friend"), ("colleague", "coworker"), ("employer", "boss"), ("supervisor", "boss"), ("bicycle", "bike"), ("refrigerator", "fridge"),
    ("television", "tv"), ("laboratory", "lab"), ("photograph", "photo"), ("advertisement", "ad"), ("examination", "exam"), ("mathematics", "math"), ("gymnasium", "gym"), ("influenza", "flu"),
    ("memorandum", "memo"), ("veterinarian", "vet"), ("sofa", "couch"), ("rubbish", "trash"), ("garbage", "trash"), ("lift", "elevator"), ("petrol", "gasoline"), ("biscuit", "cookie"),
    ("pavement", "sidewalk"), ("torch", "flashlight"), ("autumn", "fall"), ("holiday", "vacation"), ("trousers", "pants"), ("jumper", "sweater"), ("nappy", "diaper"), ("flat", "apartment"),
    ("lorry", "truck"), ("chips", "fries"), ("aubergine", "eggplant"), ("courgette", "zucchini"), ("coriander", "cilantro"), ("football", "soccer"), ("queue", "line"), ("tap", "faucet"),
    ("wardrobe", "closet"), ("cinema", "movie"), ("film", "movie"), ("ocean", "sea"), ("stream", "river"), ("creek", "brook"), ("forest", "woods"), ("road", "street"), ("path", "trail"),
    ("boat", "ship"), ("rock", "stone"), ("maize", "corn"), ("lad", "boy"), ("lass", "girl"), ("lady", "woman"), ("chap", "guy"), ("mate", "friend"), ("pal", "buddy"), ("cash", "money"),
    ("wealth", "riches"), ("salary", "pay"), ("wage", "pay"), ("task", "chore"), ("error", "mistake"), ("blunder", "mistake"), ("fault", "flaw"), ("glitch", "bug"), ("speech", "talk"),
    ("conversation", "chat"), ("argument", "fight"), ("quarrel", "fight"), ("smile", "grin"), ("tummy", "belly"), ("medicine", "drug"), ("remedy", "cure"), ("illness", "sickness"),
    ("ailment", "disease"), ("slumber", "sleep"), ("dinner", "supper"), ("feast", "banquet"), ("cuisine", "food"), ("residence", "dwelling"), ("physician", "medic"), ("automobile", "vehicle"),
    ("hat", "cap"), ("pail", "bucket"), ("sack", "bag"), ("spade", "shovel"), ("tub", "bath"), ("duvet", "quilt"), ("settee", "sofa"), ("cot", "crib"), ("kerb", "curb"), ("tyre", "tire"),
    ("jail", "prison"), ("cop", "policeman"), ("doc", "doctor"),
    ("trainers", "sneakers"), ("raincoat", "mac"), ("handbag", "purse"), ("jacket", "coat"), ("dustbin", "trash"), ("windscreen", "windshield"), ("motorway", "highway"),
    ("sweets", "candy"), ("cooker", "stove"), ("rubber", "eraser"), ("plaster", "bandage"), ("chemist", "pharmacy"), ("post", "mail"), ("postman", "mailman"), ("solicitor", "lawyer"),
    ("timetable", "schedule"), ("trolley", "cart"), ("shop", "store"), ("puddle", "pool"), ("cottage", "cabin"), ("mansion", "palace"), ("hut", "shack"), ("lane", "alley"),
    ("pond", "lake"), ("meadow", "field"), ("woodland", "forest"), ("canoe", "kayak"), ("sofa", "settee"), ("pram", "stroller"), ("stairs", "steps"), ("cellar", "basement")]


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sample_pairs(pool: list[str], n: int, *, max_use: int, seed: int) -> list[tuple[str, str]]:
    """n distinct unordered pairs of distinct words from `pool`, each word used at most `max_use` times (deterministic)."""
    rng = random.Random(seed)
    cand = list(itertools.combinations(sorted(set(pool)), 2))
    rng.shuffle(cand)
    use, out = {}, []
    for a, b in cand:
        if use.get(a, 0) < max_use and use.get(b, 0) < max_use:
            out.append((a, b))
            use[a] = use.get(a, 0) + 1
            use[b] = use.get(b, 0) + 1
        if len(out) == n:
            break
    return out


def d_of(q, pair, templates) -> float:
    return endpoint_distance(q, pair, templates)


def analyze(fams: dict, dist: dict, ctrl_match: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """fams: family -> {"trials": [...], "pairs": [...], "n_planned": int} for persona, synonym, control, countries."""
    dbar = {f: pair_dbar(v["trials"]) for f, v in fams.items()}
    sig = {f: {k: v for k, v in family_signature(fams[f]["trials"], fams[f]["n_planned"]).items() if k != "_D"} for f in fams}
    d = lambda f, p: dist[f][f"{p[0]}|{p[1]}"]
    # matched persona-control differences (the matching is by pair)
    diffs = np.array([dbar["persona"][tuple(m["persona"])] - dbar["control"][tuple(m["control"])] for m in ctrl_match
                      if tuple(m["persona"]) in dbar["persona"] and tuple(m["control"]) in dbar["control"]])
    p_matched = float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    # regression D ~ d + persona + synonym (control as reference) on the three non-country families
    rows = [(d(f, p), 1.0 if f == "persona" else 0.0, 1.0 if f == "synonym" else 0.0, v) for f in ("persona", "synonym", "control") for p, v in dbar[f].items()]
    X = np.array([[1.0, a, b, c] for a, b, c, _ in rows])
    y = np.array([r[3] for r in rows])
    fit = lambda idx: np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    beta = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    B = np.array([fit(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    ci = lambda j: [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
    med_ctrl = float(np.median(list(dbar["control"].values())))
    margin = 0.25 * med_ctrl
    persona_equiv = bool(ci(2)[0] > -margin and ci(2)[1] < margin)
    allp = [(d(f, p), v) for f in dbar for p, v in dbar[f].items()]
    out = {"signature": sig, "n_pairs": {f: len(v) for f, v in dbar.items()}, "median_dbar": {f: float(np.median(list(v.values()))) for f, v in dbar.items()},
           "median_d": {f: float(np.median([d(f, p) for p in v])) for f, v in dbar.items()},
           "matched_persona_minus_control": {"n": int(len(diffs)), "median_diff": float(np.median(diffs)), "wilcoxon_two_sided_p": p_matched},
           "regression": {"intercept": float(beta[0]), "distance": {"coef": float(beta[1]), "ci95": ci(1)}, "persona": {"coef": float(beta[2]), "ci95": ci(2)},
                          "synonym": {"coef": float(beta[3]), "ci95": ci(3)}, "equivalence_margin": margin, "persona_equivalent_to_control": persona_equiv},
           "spearman_d_vs_dbar": {"all_pairs": float(stats.spearmanr(*zip(*allp))[0]),
                                  **{f: float(stats.spearmanr([d(f, p) for p in v], list(v.values()))[0]) for f, v in dbar.items()}}}
    persona_effect = bool(ci(2)[0] > 0 and p_matched < 0.01 and float(np.median(diffs)) > 0)
    out["persona_effect_beyond_distance"] = persona_effect
    out["distance_explains"] = bool(out["spearman_d_vs_dbar"]["all_pairs"] > 0.3 and not persona_effect)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pairs":
        import torch

        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        cfg = load_ignition("data/experiments/ignition.json")
        templates = cfg.noun_ctx_templates[CARRIERS]
        ok = lambda w: len(q.tokenizer.encode(" " + w, add_special_tokens=False)) == 1
        persona_pool = [w for w in PERSONA_POOL if ok(w)]
        object_pool = [w for w in OBJECT_POOL if ok(w)]
        persona = sample_pairs(persona_pool, N, max_use=4, seed=SEED)
        cand = sample_pairs(object_pool, 400, max_use=8, seed=SEED + 1)
        synonym = [p for p in SYNONYM_PAIRS if ok(p[0]) and ok(p[1])][:N]
        singles = [c for c in cfg.countries_12 if ok(c)]
        countries = list(itertools.combinations(singles, 2))
        dist = {"persona": {}, "synonym": {}, "candidate": {}, "countries": {}}
        for fam, pairs in (("persona", persona), ("synonym", synonym), ("candidate", cand), ("countries", countries)):
            for p in pairs:
                dist[fam][f"{p[0]}|{p[1]}"] = d_of(q, p, templates)
            print(fam, len(pairs), flush=True)
        rows = match({tuple(k.split("|")): v for k, v in dist["persona"].items()}, {tuple(k.split("|")): v for k, v in dist["candidate"].items()})
        dist["control"] = {f"{r['control'][0]}|{r['control'][1]}": r["d_control"] for r in rows}
        res = {"prereg_sha256": prereg_sha256(), "layer": 24, "n_pool": {"persona": len(persona_pool), "object": len(object_pool)}, "distances": dist, "matching": rows,
               "quality": match_quality(rows),
               "pairs": {"persona": persona, "synonym": synonym, "control": [list(r["control"]) for r in rows], "countries": countries}}
        json.dump(res, open(PAIRS, "w"))
        print(json.dumps({"n": {k: len(v) for k, v in res["pairs"].items()}, "quality": res["quality"],
                          "median_d": {f: float(np.median(list(dist[f].values()))) for f in ("persona", "synonym", "control", "countries")}}, indent=1))
    elif mode == "run":
        import torch

        from hlens.loading import load_qwen

        P = json.load(open(PAIRS))
        fams = {f: [tuple(p) for p in P["pairs"][f]] for f in ("persona", "synonym", "control", "countries")}
        q = load_qwen(dtype=torch.float32)
        res = run(q, families=fams, carriers=CARRIERS, n_pairs=N, checkpoint=CKPT)
        os.makedirs("results", exist_ok=True)
        json.dump({"prereg_sha256": prereg_sha256(), "families": res}, open(OUT, "w"))
    else:
        P = json.load(open(PAIRS))
        res = json.load(open(OUT))
        # the countries family has no cap of N on pairs; run() was given n_pairs=N >= 66
        out = analyze(res["families"], P["distances"], P["matching"])
        out["matching_quality"] = P["quality"]
        out["prereg_sha256"] = {"recorded": res["prereg_sha256"], "pairs_file": P["prereg_sha256"], "current": prereg_sha256()}
        res["analysis"] = out
        json.dump(res, open(OUT, "w"))
        print(json.dumps(out, indent=1))
