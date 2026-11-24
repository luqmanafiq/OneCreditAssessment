"""The summary format, and v0: a summary written from the facts by a template, with no model.

v0 is the baseline every model-written version has to beat. It's also a check on the evaluator: it can only state
numbers from the facts, so it should score 100% on faithfulness.
"""

from summariser.facts import Facts, Flag, rm

# The format every version is asked to follow. It lives here so the prompts and the template can't drift apart.
FORMAT_SPEC = """\
Write in Markdown, in this order:

1. `**Verdict: <High concern | Look closer | Low concern>** — <one sentence on why>`
   - High concern: something that could stop the deal (not Active, winding-up petition, bad audit opinion,
     negative equity, or a situation about to become one of these).
   - Look closer: real questions to ask before going ahead (losses, falling revenue or profit, material open
     litigation, missing accounts, a possible change of control).
   - Low concern: nothing material found.
2. `**Red flags**` then one bullet per flag, most serious first, each with the number that backs it up. If there
   are none, write `**Red flags:** none found.` on one line.
3. `**Snapshot**` then two or three sentences: what the company does, where, how old; size and direction of revenue
   and profit; who owns it.
4. Optionally `**Also worth knowing**` with up to three short bullets for minor points.

Rules: under 160 words. Plain English for a busy reader, no filler, no advice beyond the verdict. Money as
RM 1.23m / RM 456k. Never state a number that isn't in the input. If something isn't in the input, don't guess.
"""

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _sorted(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: SEVERITY_ORDER[f.severity])


def _reason(facts: Facts) -> str:
    serious = [f for f in _sorted(facts.flags) if f.severity != "info"]
    if not serious:
        return "no red flags in the record."
    first = serious[0].text.rstrip(".")
    more = len(serious) - 1
    return first[0].lower() + first[1:] + (f", plus {more} more." if more else ".")


def _snapshot(facts: Facts) -> str:
    parts = [
        f"{facts.nature_of_business} in {facts.state}, incorporated {facts.incorporated[:4]} "
        f"({facts.age_years:g} years), status {facts.status}."
    ]
    if facts.years:
        last = facts.years[-1]
        line = f"FY{last['year']} revenue {rm(last['revenue'])}, profit after tax {rm(last['profit_after_tax'])}"
        if len(facts.years) >= 2:
            first = facts.years[0]
            line += f" (FY{first['year']}: {rm(first['revenue'])} and {rm(first['profit_after_tax'])})"
        parts.append(line + ".")
    else:
        parts.append("No financial statements on record.")
    parts.append(f"Largest shareholder: {facts.shareholders[0]}." if facts.shareholders else "No shareholders listed.")
    return " ".join(parts)


def render(facts: Facts) -> str:
    serious = [f for f in _sorted(facts.flags) if f.severity != "info"]
    minor = [f for f in facts.flags if f.severity == "info"][:3]
    lines = [f"**Verdict: {facts.verdict}** — {_reason(facts)}", ""]
    if serious:
        lines += ["**Red flags**", *[f"- {f.text}" for f in serious], ""]
    else:
        lines += ["**Red flags:** none found.", ""]
    lines += ["**Snapshot**", _snapshot(facts)]
    if minor:
        lines += ["", "**Also worth knowing**", *[f"- {f.text}" for f in minor]]
    return "\n".join(lines)
