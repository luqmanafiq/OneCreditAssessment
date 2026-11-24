"""Command line: python -m summariser {facts,generate,eval}.

facts                          write runs/facts.jsonl (what the rules found, for reading)
generate VERSION [--labelled] [--only SYN-...] [--dry-run]
    VERSION is v0-template, v1-raw or v2-grounded. --labelled runs only the 22 hand-labelled companies, to save
    free-tier quota. --dry-run saves the prompts without calling the model.
eval                           score every summaries/*.jsonl, write eval/results.md
"""

import argparse
import json

from summariser import evaluate, prompts
from summariser.facts import ROOT, extract_all, load_records
from summariser.llm import RUNS, Client
from summariser.render import render

VERSIONS = ("v0-template", "v1-raw", "v2-grounded")


def generate(version: str, labelled: bool, only: list[str], dry_run: bool) -> None:
    records = {r["registrationNumber"]: r for r in load_records()}
    facts = extract_all()
    if labelled:
        keep = set(evaluate.load_labels())
        facts = [f for f in facts if f.registration_number in keep]
    if only:
        facts = [f for f in facts if f.registration_number in only]
    client = None if version == "v0-template" else Client(version, dry_run=dry_run)
    rows = []
    for i, f in enumerate(facts, 1):
        reg = f.registration_number
        if client is None:
            text = render(f)
        else:
            messages = prompts.v1_raw(records[reg]) if version == "v1-raw" else prompts.v2_grounded(f)
            print(f"[{i}/{len(facts)}] {reg}")
            text = client.complete(reg, messages)
        rows.append({"registrationNumber": reg, "summary": text})
    if dry_run:
        print(f"Prompts saved to runs/{version}/")
        return
    out = ROOT / "summaries" / f"{version}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} summaries to {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="summariser")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("facts")
    g = sub.add_parser("generate")
    g.add_argument("version", choices=VERSIONS)
    g.add_argument("--labelled", action="store_true")
    g.add_argument("--only", nargs="*", default=[])
    g.add_argument("--dry-run", action="store_true")
    sub.add_parser("eval")
    args = parser.parse_args()
    if args.command == "facts":
        RUNS.mkdir(exist_ok=True)
        path = RUNS / "facts.jsonl"
        path.write_text("\n".join(json.dumps(f.to_dict(), ensure_ascii=False) for f in extract_all()) + "\n", encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")
    elif args.command == "generate":
        generate(args.version, args.labelled, args.only, args.dry_run)
    else:
        evaluate.main()


if __name__ == "__main__":
    main()
