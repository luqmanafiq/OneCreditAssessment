"""Prompts for the model-written versions.

v1-raw:      the whole record goes to the model, and it decides what matters. The obvious first thing to try.
v2-grounded: the model gets the checked facts and flags from facts.py, and only has to write. It may disagree with
             the rule-based verdict, but has to say why.
"""

import json
from typing import Any

from summariser.facts import Facts
from summariser.render import FORMAT_SPEC

ROLE = """\
You write one-screen summaries of Malaysian companies for a team that vets suppliers and project partners. They read
your summary first to decide whether a company is worth a closer look, so missing a red flag is the worst mistake you
can make, and inventing one is the second worst. The records are as of 31 December 2025.
"""


def v1_raw(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROLE + "\n" + FORMAT_SPEC},
        {
            "role": "user",
            "content": "Summarise this company record. Ratios are decimals (0.12 is 12%); amounts are RM.\n\n"
            + json.dumps(record, ensure_ascii=False),
        },
    ]


V2_INSTRUCTIONS = """\
You get facts that were already extracted and checked by code, including a list of flags with a severity and a
rule-based verdict.

- Every flag with severity "critical" or "warning" must appear under Red flags, with its number. Don't drop or merge
  them away.
- "info" flags are optional; use the most useful ones under Also worth knowing.
- Start from the rule-based verdict. You may move it up or down one level only if the figures clearly support it
  (for example, equity about to turn negative); if you do, the verdict sentence must say why.
- Use only numbers that appear in the facts. Don't compute new ones.
"""


def v2_grounded(facts: Facts) -> list[dict[str, str]]:
    payload = facts.to_dict()
    payload["rule_based_verdict"] = payload.pop("verdict")
    return [
        {"role": "system", "content": ROLE + "\n" + FORMAT_SPEC + "\n" + V2_INSTRUCTIONS},
        {"role": "user", "content": "Facts:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)},
    ]
