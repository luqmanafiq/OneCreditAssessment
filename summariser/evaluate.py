"""Score every summaries/<version>.jsonl without calling a model.

Metrics, in order of how much they matter to the vetting team:
  1. Flag recall      - of the critical/warning flags found by facts.py, how many does the summary mention?
  2. Missed "High"    - labelled High concern, but the summary's verdict isn't. The costliest error.
  3. False alarms     - the summary raises a serious red flag (winding-up, bad audit, negative equity) that isn't there.
  4. Faithfulness     - of the RM amounts, percentages and multiples in the summary, how many match the record?
  5. Verdict accuracy - against hand labels in labels/verdicts.csv (dev and held-out reported separately).
  6. Length and format.

The checks are regexes, so they're cheap, repeatable and explainable, but they can be fooled by wording they don't
expect. Mismatches are written to eval/details/<version>.jsonl so they can be read by eye.
"""

import csv
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from summariser.facts import ROOT, Facts, extract_all, load_records

SUMMARIES = ROOT / "summaries"
LABELS = ROOT / "labels" / "verdicts.csv"
OUT = ROOT / "eval"

# How a summary can mention each flag. Deliberately loose: recall should fail only when the point is really missing.
MENTIONS: dict[str, str] = {
    "status": r"status[^.\n]{0,40}winding|winding[- ]up[^.\n]{0,20}status|not active|(in|under|being) (winding|wound)",
    "winding_up_petition": r"winding[- ]up petition|petition",
    "audit_opinion": r"disclaim|going[- ]concern|(?<!un)qualified|adverse opinion",
    "negative_equity": r"negative (equity|net (worth|assets))|liabilities exceed"
    r"|equity[^.\n]{0,30}(negative|deficit|-RM|\u2212RM)",
    "open_litigation": r"pending|judgment|lawsuit|litigation|claims?\b|cases?\b",
    "no_financials": r"no financial|financials?[^.\n]{0,20}(missing|unavailable|not available)|no (accounts|statements)",
    "missing_year": r"FY ?20\d\d[^.\n]{0,50}(missing|gap|not filed|absent|no financ)"
    r"|(missing|gap|no financ|no accounts)",
    "persistent_losses": r"loss",
    "latest_loss": r"loss",
    "profit_decline": r"profit[^.\n]{0,80}(fell|fall|drop|declin|down|shr[ui]nk)"
    r"|(fell|fall|drop|declin|down)[^.\n]{0,40}profit",
    "revenue_decline": r"(revenue|sales|turnover)[^.\n]{0,80}(fell|fall|drop|declin|down|shr[ui]nk)"
    r"|(fell|fall|drop|declin|down|shr[ui]nk)[^.\n]{0,40}(revenue|sales|turnover)",
    "repeated_winding_up": r"(\d|two|three|four|several|multiple|repeated)[^.\n]{0,30}petitions",
    "large_settled_claim": r"settle",
    "control_change": r"control|ownership[^.\n]{0,20}chang|new (majority|owner)",
}

# Serious claims a summary must not make unless the record backs them. Sentences with a negation are skipped.
ALARMS: dict[str, tuple[str, set[str]]] = {
    "winding-up": (r"winding[- ]up", {"status", "winding_up_petition", "repeated_winding_up"}),
    "audit": (r"disclaim|going[- ]concern|(?<!un)qualified opinion|adverse opinion", {"audit_opinion"}),
    "negative equity": (r"negative (equity|net (worth|assets))", {"negative_equity"}),
}
NEGATION = re.compile(r"\b(no|none|not|without|never)\b", re.IGNORECASE)

VERDICT = re.compile(r"\*\*\s*Verdict:\s*(High concern|Look closer|Low concern)", re.IGNORECASE)
LEVELS = {"high concern": "High concern", "look closer": "Look closer", "low concern": "Low concern"}

AMOUNT = re.compile(r"[-\u2212]?\s?RM\s?([\d,]+(?:\.\d+)?)\s?(bn|billion|b|m|mil|million|k)?\b", re.IGNORECASE)
PERCENT = re.compile(r"([\d,]+(?:\.\d+)?)\s?%")
MULTIPLE = re.compile(r"\b(\d+(?:\.\d+)?)\s?x\b")
UNITS = {"": 1, "k": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def _numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers(v)]
    return []


def _parse(text: str) -> list[tuple[str, float, float]]:
    """(kind, value, tolerance) for every number in the text. Tolerance is half the last displayed digit."""
    found: list[tuple[str, float, float]] = []
    for m in AMOUNT.finditer(text):
        digits, unit = m.group(1).replace(",", ""), UNITS[(m.group(2) or "").lower()]
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(("RM", float(digits) * unit, 0.5 * 10**-decimals * unit))
    for m in PERCENT.finditer(text):
        digits = m.group(1).replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(("%", float(digits), 0.5 * 10**-decimals))
    for m in MULTIPLE.finditer(text):
        digits = m.group(1)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(("x", float(digits), 0.5 * 10**-decimals))
    return found


def allowed_numbers(record: dict[str, Any], facts: Facts) -> dict[str, list[float]]:
    """Every number a summary may state: anything in the record, plus what facts.py derived from it."""
    raw = [abs(n) for n in _numbers(record)]
    derived = [(k, abs(v)) for f in facts.flags for k, v, _ in _parse(f.text)]
    open_total = facts.cases["open_against_total"]
    return {
        "RM": raw + [v for k, v in derived if k == "RM"] + [open_total],
        "%": [n * 100 for n in raw] + raw + [v for k, v in derived if k == "%"],
        "x": raw + [v for k, v in derived if k == "x"],
    }


def _supported(kind: str, value: float, tol: float, allowed: dict[str, list[float]]) -> bool:
    return any(abs(value - a) <= max(tol, 0.005 * a) + 1e-9 for a in allowed[kind])


def load_labels() -> dict[str, dict[str, str]]:
    with LABELS.open(encoding="utf-8") as f:
        return {row["registration_number"]: row for row in csv.DictReader(f)}


def score(version: str, summaries: dict[str, str]) -> dict[str, Any]:
    records = {r["registrationNumber"]: r for r in load_records()}
    facts = {f.registration_number: f for f in extract_all()}
    labels = load_labels()
    required = mentioned = alarms = nums = ok_nums = over_length = no_verdict = 0
    missed_by_flag: Counter[str] = Counter()
    verdicts: dict[str, list[tuple[str, str | None]]] = {"dev": [], "heldout": []}
    details = []
    for reg, text in summaries.items():
        fx = facts[reg]
        flag_ids = {f.id for f in fx.flags}
        row: dict[str, Any] = {"registrationNumber": reg}
        # 1. Flag recall
        needed = sorted({f.id for f in fx.flags if f.severity != "info"})
        missing = [i for i in needed if not re.search(MENTIONS[i], text, re.IGNORECASE)]
        required += len(needed)
        mentioned += len(needed) - len(missing)
        missed_by_flag.update(missing)
        row["missed_flags"] = missing
        # 3. False alarms
        sentences = re.split(r"(?<=[.!?])\s+|\n", text)
        raised = [
            name
            for name, (pattern, backing) in ALARMS.items()
            if not flag_ids & backing
            and any(re.search(pattern, s, re.IGNORECASE) and not NEGATION.search(s) for s in sentences)
        ]
        alarms += bool(raised)
        row["false_alarms"] = raised
        # 4. Faithfulness
        allowed = allowed_numbers(records[reg], fx)
        parsed = _parse(text)
        bad = [f"{v:g}{k}" for k, v, t in parsed if not _supported(k, v, t, allowed)]
        nums += len(parsed)
        ok_nums += len(parsed) - len(bad)
        row["unsupported_numbers"] = bad
        # 5. Verdict
        m = VERDICT.search(text)
        predicted = LEVELS[m.group(1).lower()] if m else None
        no_verdict += predicted is None
        row["verdict"] = predicted
        if reg in labels:
            verdicts[labels[reg]["split"]].append((labels[reg]["verdict"], predicted))
            row["label"] = labels[reg]["verdict"]
        # 6. Length
        words = len(re.findall(r"\b\w[\w.,%-]*\b", text))
        over_length += words > 160
        row["words"] = words
        details.append(row)

    labelled = verdicts["dev"] + verdicts["heldout"]
    OUT.joinpath("details").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("details", f"{version}.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n", encoding="utf-8"
    )
    return {
        "version": version,
        "n": len(summaries),
        "flag_recall": mentioned / required if required else 1.0,
        "flags": f"{mentioned}/{required}",
        "missed_by_flag": dict(missed_by_flag.most_common()),
        "missed_high": sum(gold == "High concern" and pred != "High concern" for gold, pred in labelled),
        "labelled_high": sum(gold == "High concern" for gold, _ in labelled),
        "false_alarm_companies": alarms,
        "faithfulness": ok_nums / nums if nums else 1.0,
        "numbers": f"{ok_nums}/{nums}",
        "verdict_dev": _accuracy(verdicts["dev"]),
        "verdict_heldout": _accuracy(verdicts["heldout"]),
        "no_verdict": no_verdict,
        "median_words": sorted(d["words"] for d in details)[len(details) // 2] if details else 0,
        "over_160_words": over_length,
    }


def _accuracy(pairs: Sequence[tuple[str, str | None]]) -> str:
    if not pairs:
        return "-"
    return f"{sum(g == p for g, p in pairs)}/{len(pairs)}"


def rules_baseline() -> dict[str, str]:
    """How well the rule-based verdict alone matches the hand labels: the bar a model's judgement has to clear."""
    facts = {f.registration_number: f for f in extract_all()}
    labels = load_labels()
    out = {}
    for split in ("dev", "heldout"):
        pairs = [(row["verdict"], facts[reg].verdict) for reg, row in labels.items() if row["split"] == split]
        out[split] = _accuracy(pairs)
    return out


def read_version(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["registrationNumber"]: r["summary"] for r in rows if r.get("summary")}


def main() -> None:
    results = [score(p.stem, read_version(p)) for p in sorted(SUMMARIES.glob("*.jsonl"))]
    baseline = rules_baseline()
    lines = [
        "# Evaluation results",
        "",
        "Generated by `just eval`. Verdict accuracy is against the hand labels in `labels/verdicts.csv`. "
        f"For reference, the rule-based verdict alone gets {baseline['dev']} on dev and {baseline['heldout']} on held-out.",
        "",
        "| Version | n | Flag recall | Missed High | False alarms | Faithful numbers | Verdict (dev) | Verdict (held-out)"
        " | Median words | >160 words |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['version']} | {r['n']} | {r['flag_recall']:.0%} ({r['flags']}) | {r['missed_high']}/{r['labelled_high']}"
        f" | {r['false_alarm_companies']} | {r['faithfulness']:.0%} ({r['numbers']}) | {r['verdict_dev']}"
        f" | {r['verdict_heldout']} | {r['median_words']} | {r['over_160_words']} |"
        for r in results
    )
    lines += ["", "Flags most often missed:", ""]
    lines += [f"- {r['version']}: {r['missed_by_flag'] or 'none'}" for r in results]
    OUT.mkdir(exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
