"""Word error rate (WER): the share of words the listener got wrong.

WER = (substituted + deleted + inserted words) / words in the reference.
Before comparing, both texts are normalised so "4 p.m." and "four pm" count
as the same thing.
"""

from __future__ import annotations

import re

import jiwer

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Spell out 0-9999 the way people say it ("eight thirty" style is handled
    separately for clock times)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        return _ONES[thousands] + " thousand" + ("" if rest == 0 else " " + number_to_words(rest))
    return " ".join(_ONES[int(d)] for d in str(n))


def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)  # 7pm -> 7 pm
    t = re.sub(r"\b([ap])\.?\s?m\b\.?", r"\1m", t)  # p.m. / p m -> pm
    t = re.sub(r"\b(\d{1,2})([0-5]\d) (am|pm)\b", r"\1:\2 \3", t)  # 830 pm -> 8:30 pm
    # clock times: 8:30 or 8.30 -> eight thirty, 8:00 -> eight
    t = re.sub(
        r"\b(\d{1,2})[:.](\d{2})\b",
        lambda m: (
            number_to_words(int(m.group(1)))
            + ("" if m.group(2) == "00" else " " + number_to_words(int(m.group(2))))
        ),
        t,
    )
    t = re.sub(r"\b\d+\b", lambda m: number_to_words(int(m.group(0))), t)
    t = t.replace("o'clock", "").replace("-", " ")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(reference: str, hypothesis: str) -> float:
    """WER between what was said and what was heard, after normalising.
    Returns 0.0 for a perfect match; can exceed 1.0 with many insertions."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))
