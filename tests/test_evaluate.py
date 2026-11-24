"""Does the evaluator catch what it claims to? Hand-written summaries, phrased the way a model might phrase them."""

from summariser.evaluate import score

GOOD_545 = """**Verdict: High concern** — the company is being wound up.

**Red flags**
- Status is Winding up; a winding-up petition ended in judgment (RM 10k).
- Auditor flagged material uncertainty over going concern in FY2022.
- Five cases against it are pending or lost, with claims of RM 2.51m.
- Two winding-up petitions since 2023.
- New directors appointed in 2025 hold 84.6% of shares, so control may have changed.

**Snapshot**
Accounting software firm in Selangor. Revenue RM 13.01m in FY2024."""

# Same company, but drops the audit and litigation points and invents a number.
BAD_545 = """**Verdict: Look closer** — the company is in winding up status.

**Red flags**
- Status is Winding up after a petition.
- Two winding-up petitions since 2023; new owners took control in 2025.

**Snapshot**
Revenue was RM 14.2m in FY2024 with a 9.9% margin."""

# A clean company described with a red flag that isn't there, and a negated mention that is fine.
ALARM_1645 = """**Verdict: Look closer** — the auditor issued a disclaimer of opinion.

**Red flags:** none found. There are no winding-up petitions.

**Snapshot**
Fit-out contractor, revenue RM 21.29m in FY2024, profit RM 931k."""


def test_good_summary_scores_full_marks() -> None:
    r = score("test-good", {"SYN-000545": GOOD_545})
    assert r["flag_recall"] == 1.0, r["missed_by_flag"]
    assert r["false_alarm_companies"] == 0
    assert r["faithfulness"] == 1.0
    assert r["missed_high"] == 0


def test_missing_flags_and_invented_numbers_are_caught() -> None:
    r = score("test-bad", {"SYN-000545": BAD_545})
    assert set(r["missed_by_flag"]) == {"audit_opinion", "open_litigation"}
    assert r["numbers"] == "0/2"  # RM 14.2m and 9.9% are not in the record


def test_missed_high_concern_is_counted() -> None:
    # SYN-001212 is hand-labelled High concern (status Winding up).
    r = score("test-missed", {"SYN-001212": "**Verdict: Look closer** — status is Winding up after a petition."})
    assert r["missed_high"] == 1


def test_false_alarm_caught_but_negation_ignored() -> None:
    r = score("test-alarm", {"SYN-001645": ALARM_1645})
    assert r["false_alarm_companies"] == 1
    assert r["faithfulness"] == 1.0


def test_missing_verdict_line_counts() -> None:
    r = score("test-noverdict", {"SYN-001645": "A profitable contractor with revenue of RM 21.29m."})
    assert r["no_verdict"] == 1
