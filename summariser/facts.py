"""Turn a raw company record into checked facts and red flags, without a model.

Everything a summary is allowed to claim starts here. The rules are deliberately simple and each flag carries the
numbers behind it, so a reader (or the evaluator) can trace every statement back to the record.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "data" / "companies.jsonl"
AS_OF = date(2025, 12, 31)  # the records are a snapshot as of this date (README)

Severity = Literal["critical", "warning", "info"]
Verdict = Literal["High concern", "Look closer", "Low concern"]

# Statuses of litigation where the company still owes, or may owe, money.
OPEN_AGAINST = {"Pending", "Judgment entered"}


@dataclass
class Flag:
    id: str  # stable key, used by the evaluator
    severity: Severity
    text: str  # one plain sentence, with the numbers that back it up


@dataclass
class Facts:
    registration_number: str
    name: str
    industry: str
    state: str
    status: str
    nature_of_business: str
    incorporated: str
    age_years: float
    directors: int
    shareholders: list[str]
    years: list[dict[str, Any]]  # trimmed per-year figures, oldest first
    cases: dict[str, Any]
    flags: list[Flag] = field(default_factory=list)
    verdict: Verdict = "Low concern"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in COMPANIES.read_text(encoding="utf-8").splitlines() if line.strip()]


def rm(amount: float) -> str:
    """Format RM amounts the way a human reader scans them: RM 1.23m, RM 456k, RM 900."""
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    if a >= 1_000_000:
        return f"{sign}RM {a / 1_000_000:.2f}m"
    if a >= 1_000:
        return f"{sign}RM {a / 1_000:.0f}k"
    return f"{sign}RM {a:.0f}"


def pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def _years(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for y in sorted(record["financials"]["years"], key=lambda y: y["financialYear"]):
        s, r = y["statements"], y["ratiosAndMetrics"]
        out.append(
            {
                "year": y["financialYear"],
                "year_end": y["financialYearEnd"],
                "audit_opinion": y["audit"]["opinion"],
                "revenue": s["revenue"],
                "profit_after_tax": s["profitAfterTax"],
                "total_equity": s["totalEquity"],
                "borrowings": s["borrowings"],
                "operating_cash_flow": s["operatingCashFlow"],
                "cash": s["cash"],
                "net_margin": r["profitability"]["netProfitMargin"],
                "quick_ratio": r["liquidityAndCashFlow"]["quickRatio"],
                "interest_cover": r["capitalStructure"]["interestCoverageRatio"],
                "leverage": r["capitalStructure"]["leverageRatio"],
            }
        )
    return out


def _cases(record: dict[str, Any]) -> dict[str, Any]:
    cases = record["litigation"]["cases"]
    against = [c for c in cases if c["role"] == "defendant" and c["status"] in OPEN_AGAINST]
    return {
        "total": len(cases),
        "as_defendant": sum(c["role"] == "defendant" for c in cases),
        "as_plaintiff": sum(c["role"] == "plaintiff" for c in cases),
        "open_against": [
            {"type": c["caseType"], "status": c["status"], "claim": c["claimAmount"], "filed": c["filedOn"]} for c in against
        ],
        "open_against_total": sum(c["claimAmount"] for c in against),
        "winding_up_against": [c for c in against if c["caseType"] == "Winding-up petition"],
    }


def _flags(record: dict[str, Any], years: list[dict[str, Any]], cases: dict[str, Any], age: float) -> list[Flag]:
    flags: list[Flag] = []
    p = record["profile"]

    # --- Status and solvency: any one of these means "don't proceed without a hard look".
    if p["status"] != "Active":
        flags.append(Flag("status", "critical", f"Company status is '{p['status']}', not Active."))
    if cases["winding_up_against"]:
        parts = [f"{c['status'].lower()}, {rm(c['claimAmount'])}" for c in cases["winding_up_against"]]
        flags.append(
            Flag("winding_up_petition", "critical", f"Winding-up petition(s) against the company: {'; '.join(parts)}.")
        )
    for y in years:
        op = y["audit_opinion"]
        if op != "Unqualified":  # anything else (disclaimer, adverse, qualified, going-concern doubt) is a red flag
            flags.append(Flag("audit_opinion", "critical", f"FY{y['year']} audit opinion: {op}."))
    if years and years[-1]["total_equity"] < 0:
        flags.append(
            Flag("negative_equity", "critical", f"Negative equity of {rm(years[-1]['total_equity'])} in FY{years[-1]['year']}.")
        )

    # --- Litigation that may cost the company money.
    if cases["open_against"]:
        n, total = len(cases["open_against"]), cases["open_against_total"]
        equity = years[-1]["total_equity"] if years else None
        material = equity is not None and equity > 0 and total > 0.25 * equity
        extra = f" ({pct(total / equity)} of latest equity)" if equity and equity > 0 else ""
        flags.append(
            Flag(
                "open_litigation",
                "warning" if material or n >= 3 else "info",
                f"{n} case(s) against the company still pending or with judgment entered, claims totalling {rm(total)}{extra}.",
            )
        )

    # Added after labelling the dev sample (SYN-001389): repeated petitions show creditor pressure even when settled.
    petitions = [
        c for c in record["litigation"]["cases"] if c["role"] == "defendant" and c["caseType"] == "Winding-up petition"
    ]
    if len(petitions) >= 2:
        flags.append(
            Flag(
                "repeated_winding_up",
                "warning",
                f"{len(petitions)} winding-up petitions filed against the company since "
                f"{min(c['filedOn'] for c in petitions)[:4]}.",
            )
        )
    # Added after labelling (SYN-000397): a settlement can still cost money, and the amount isn't in the record.
    equity_now = years[-1]["total_equity"] if years else None
    flags.extend(
        Flag(
            "large_settled_claim",
            "warning",
            f"Settled a {c['caseType'].lower()} claim of {rm(c['claimAmount'])} filed {c['filedOn']}, "
            f"larger than equity of {rm(equity_now)}; settlement terms unknown.",
        )
        for c in record["litigation"]["cases"]
        if c["role"] == "defendant" and c["status"] == "Settled" and equity_now and c["claimAmount"] > equity_now
    )

    # --- Financial record: missing, thin or deteriorating.
    if not years:
        flags.append(Flag("no_financials", "warning", "No financial statements on record."))
    else:
        nums = [y["year"] for y in years]
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        if missing:
            flags.append(
                Flag(
                    "missing_year",
                    "warning",
                    f"No financials for FY{', FY'.join(map(str, missing))} (gap in the filing record).",
                )
            )
        last = years[-1]
        losses = [y for y in years if y["profit_after_tax"] < 0]
        if len(years) >= 2 and len(losses) == len(years):
            flags.append(
                Flag(
                    "persistent_losses",
                    "warning",
                    f"Loss-making in every year on record; latest loss {rm(last['profit_after_tax'])} in FY{last['year']}.",
                )
            )
        elif last["profit_after_tax"] < 0:
            flags.append(Flag("latest_loss", "warning", f"Loss of {rm(last['profit_after_tax'])} in FY{last['year']}."))
        if len(years) >= 2 and not losses:
            first = years[0]
            if first["profit_after_tax"] > 0 and last["profit_after_tax"] < 0.5 * first["profit_after_tax"]:
                drop = 1 - last["profit_after_tax"] / first["profit_after_tax"]
                flags.append(
                    Flag(
                        "profit_decline",
                        "warning",
                        f"Profit after tax fell {pct(drop)}, from {rm(first['profit_after_tax'])} (FY{first['year']}) "
                        f"to {rm(last['profit_after_tax'])} (FY{last['year']}).",
                    )
                )
        # Added after labelling (SYN-000923): shrinking revenue is the earliest sign of a customer leaving.
        if len(years) >= 2 and years[0]["revenue"] > 0 and last["revenue"] < 0.8 * years[0]["revenue"]:
            fall = 1 - last["revenue"] / years[0]["revenue"]
            flags.append(
                Flag(
                    "revenue_decline",
                    "warning",
                    f"Revenue fell {pct(fall)}, from {rm(years[0]['revenue'])} (FY{years[0]['year']}) "
                    f"to {rm(last['revenue'])} (FY{last['year']}).",
                )
            )
        if len(years) >= 2 and years[0]["profit_after_tax"] < 0 < last["profit_after_tax"]:
            flags.append(
                Flag(
                    "turnaround",
                    "info",
                    f"Turned from a {rm(years[0]['profit_after_tax'])} loss (FY{years[0]['year']}) "
                    f"to a {rm(last['profit_after_tax'])} profit (FY{last['year']}).",
                )
            )
        if last["operating_cash_flow"] < 0 and last["profit_after_tax"] < 0:
            flags.append(
                Flag("cash_burn", "info", f"Operating cash flow was {rm(last['operating_cash_flow'])} in FY{last['year']}.")
            )
        ic = last["interest_cover"]
        if ic is not None and ic < 1 and last["borrowings"] > 0 and last["profit_after_tax"] >= 0:
            # Loss-makers always have cover < 1, so only flag it when it adds information.
            flags.append(Flag("thin_interest_cover", "info", f"Interest cover of {ic:.2f}x in FY{last['year']}."))
        if last["quick_ratio"] is not None and last["quick_ratio"] < 0.5:
            flags.append(Flag("low_liquidity", "info", f"Quick ratio of {last['quick_ratio']:.2f} in FY{last['year']}."))

    # --- Governance and identity: not bad in themselves, but worth knowing before a deal.
    if age < 3:
        flags.append(Flag("young_company", "info", f"Incorporated {p['incorporationDate']}, about {age:.1f} years old."))
    recent = [d for d in record["relationships"]["directors"] if d["appointedOn"] >= f"{AS_OF.year}-01-01"]
    if len(recent) >= 2 and age >= 2:
        n_dir = len(record["relationships"]["directors"])
        flags.append(Flag("director_churn", "info", f"{len(recent)} of {n_dir} directors were appointed in {AS_OF.year}."))
    # Added after labelling (SYN-000029): new directors who hold most of the shares suggest a change of control.
    if age >= 2:
        new_names = {d["name"] for d in recent}
        new_stake = sum(s["percentage"] for s in record["relationships"]["shareholders"] if s["name"] in new_names)
        if new_stake >= 50:
            flags.append(
                Flag(
                    "control_change",
                    "warning",
                    f"Directors appointed in {AS_OF.year} together hold {new_stake:.1f}% of the shares "
                    "(possible change of control).",
                )
            )
    for f in p["formerNames"]:
        changed = date.fromisoformat(f["changedOn"])
        if (AS_OF - changed).days <= 365:
            flags.append(Flag("recent_name_change", "info", f"Renamed from {f['name']} on {f['changedOn']}."))
    flags.extend(
        Flag(
            "corporate_parent",
            "info",
            f"{s['percentage']:g}% owned by {s['name']} ({s['registrationNumber']}); check the parent too.",
        )
        for s in record["relationships"]["shareholders"]
        if s["type"] == "company" and s["percentage"] >= 50
    )
    return flags


def verdict(flags: list[Flag]) -> Verdict:
    severities = [f.severity for f in flags]
    if "critical" in severities:
        return "High concern"
    if "warning" in severities:
        return "Look closer"
    return "Low concern"


def extract(record: dict[str, Any]) -> Facts:
    p = record["profile"]
    years = _years(record)
    cases = _cases(record)
    age = (AS_OF - date.fromisoformat(p["incorporationDate"])).days / 365.25
    shareholders = [
        f"{s['name']} ({s['type']}, {s['percentage']:g}%)"
        for s in sorted(record["relationships"]["shareholders"], key=lambda s: -s["percentage"])
    ]
    flags = _flags(record, years, cases, age)
    return Facts(
        registration_number=record["registrationNumber"],
        name=p["companyName"],
        industry=p["industry"],
        state=p["state"],
        status=p["status"],
        nature_of_business=p["natureOfBusiness"],
        incorporated=p["incorporationDate"],
        age_years=round(age, 1),
        directors=len(record["relationships"]["directors"]),
        shareholders=shareholders,
        years=years,
        cases=cases,
        flags=flags,
        verdict=verdict(flags),
    )


def extract_all() -> list[Facts]:
    """Extract every company, then add links between companies that share a corporate parent."""
    records = load_records()
    facts = [extract(r) for r in records]
    parents: dict[str, list[str]] = {}
    for r in records:
        for s in r["relationships"]["shareholders"]:
            if s["type"] == "company":
                parents.setdefault(s["registrationNumber"], []).append(r["registrationNumber"])
    for f, r in zip(facts, records, strict=True):
        for s in r["relationships"]["shareholders"]:
            siblings = [x for x in parents.get(s.get("registrationNumber") or "", []) if x != f.registration_number]
            if s["type"] == "company" and siblings:
                f.flags.append(Flag("shared_parent", "info", f"{s['name']} also holds shares in {', '.join(siblings)}."))
    return facts
