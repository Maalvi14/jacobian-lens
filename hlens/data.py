from dataclasses import dataclass
import json

@dataclass(frozen=True)
class LensEvalItem:
    name: str
    prompt: str
    intermediates: list[str]
    target: str | None = None


def load_lens_eval(path: str) -> list[LensEvalItem]:
    with open(path) as f:
        data = json.load(f)
    return [
        LensEvalItem(
            name=item["name"],
            prompt=item["prompt"],
            intermediates=item["intermediates"],
            target=item.get("target"),
        )
        for item in data["items"]
    ]


@dataclass(frozen=True)
class ProbeSwapItem:
    name: str
    category: str
    prompt: str
    intermediate: str
    answer: str
    swap_to: str
    swap_answer: str


def load_probe_swap(path: str) -> list[ProbeSwapItem]:
    with open(path) as f:
        data = json.load(f)
    return [ProbeSwapItem(**item) for item in data["items"]]


@dataclass(frozen=True)
class IgnitionConfig:
    countries_12: list[str]
    alt_words: list[str]
    ctx_templates: list[str]
    noun_ctx_templates: list[str]
    idiom_pairs: list[tuple[str, str]]
    scrambled_pairs: list[tuple[str, str]]


def load_ignition(path: str) -> IgnitionConfig:
    with open(path) as f:
        data = json.load(f)
    return IgnitionConfig(
        countries_12=data["countries_12"],
        alt_words=data["alt_words"],
        ctx_templates=data["ctx_templates"],
        noun_ctx_templates=data["noun_ctx_templates"],
        idiom_pairs=[tuple(p) for p in data["idiom_pairs"]],
        scrambled_pairs=[tuple(p) for p in data["scrambled_pairs"]],
    )


@dataclass(frozen=True)
class CandidatePool:
    name: str
    pool: list[str]
    proto: list[str]


@dataclass(frozen=True)
class CapacityConfig:
    block_families: list[str]
    targets_per_family: dict[str, int]
    candidate_pools: list[CandidatePool]


def load_capacity(path: str) -> CapacityConfig:
    with open(path) as f:
        data = json.load(f)
    return CapacityConfig(
        block_families=data["block_families"],
        targets_per_family=data["targets_per_family"],
        candidate_pools=[CandidatePool(**cp) for cp in data["candidate_pools"]],
    )