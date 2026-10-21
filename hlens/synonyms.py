"""Single-token synonym sets for the order-of-operations intermediates (data/evaluations
README: "numbers -> digit and word forms; operations -> symbol and word forms"), as pinned in
hlens/docs/preregistration_h3.md."""

from __future__ import annotations

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
_TENS = {30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}

OPERATIONS = {
    "multiplication": ["*", "×", "multiplication", "times", "multiply"],
    "addition": ["+", "addition", "plus", "add", "sum"],
    "subtraction": ["-", "subtraction", "minus", "subtract"],
    "division": ["/", "÷", "division", "divide", "divided"],
    "squared": ["^2", "²", "squared", "square"],  # addendum 1 to the H3 pre-registration
    "mod": ["%", "mod", "modulo", "remainder"],
}


def number_forms(n: int) -> list[str]:
    forms = [str(n)]
    if 0 <= n <= 20:
        forms.append(_UNITS[n])
    elif n in _TENS:
        forms.append(_TENS[n])
    return forms


def forms_for(key: str) -> list[str]:
    """Surface forms of an intermediate key (a number string or an operation name)."""
    return number_forms(int(key)) if key.strip().lstrip("-").isdigit() else list(OPERATIONS[key])


def single_token_ids(tok, forms: list[str]) -> dict[str, int]:
    """form -> token id for the forms that are a single token, with or without a leading space
    (the space-prefixed variant is keyed as ' form')."""
    out = {}
    for f in forms:
        for variant in (f, " " + f):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[variant] = ids[0]
    return out
