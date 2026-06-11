#!/usr/bin/env python3
"""
Command-line interface for the clinical calculator tool-use system.

Usage:
  python scripts/infer.py "Patient with DVT symptoms..."
  python scripts/infer.py --file note.txt
  echo "clinical note" | python scripts/infer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.inference import run_inference


RISK_COLOURS = {
    "low": "\033[92m",
    "moderate": "\033[93m",
    "high": "\033[91m",
    "very_high": "\033[91m",
}
RESET = "\033[0m"
BOLD = "\033[1m"


def coloured_risk(level: str) -> str:
    c = RISK_COLOURS.get(level, "")
    return f"{c}{level.upper().replace('_', ' ')}{RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical Calculator Tool-Use System CLI"
    )
    parser.add_argument("note", nargs="?", help="Clinical note text (or use --file / stdin)")
    parser.add_argument("--file", "-f", help="Path to a text file containing the clinical note")
    parser.add_argument("--model", "-m", default="claude-opus-4-5", help="Anthropic model")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    # Resolve note text
    if args.note:
        note = args.note
    elif args.file:
        note = Path(args.file).read_text()
    elif not sys.stdin.isatty():
        note = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    note = note.strip()
    if not note:
        print("Error: empty clinical note", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    print(f"\n{BOLD}Analysing clinical note...{RESET}\n")

    result = run_inference(note, api_key=api_key, model=args.model)

    if args.json:
        print(json.dumps({
            "calculator_results": [r.to_dict() for r in result.calculator_results],
            "reasoning": result.reasoning,
            "extracted_parameters": result.extracted_parameters,
            "model_used": result.model_used,
        }, indent=2))
        return

    # ── Formatted output ────────────────────────────────────────────────────
    if not result.calculator_results:
        print("⚠  No calculators were applied.")
        if result.reasoning:
            print(f"\n{result.reasoning}")
        return

    for r in result.calculator_results:
        print(f"{'─' * 60}")
        print(f"{BOLD}▶ {r.calculator}{RESET}")
        print(f"  Score          : {BOLD}{r.score}{RESET}")
        print(f"  Risk level     : {coloured_risk(r.risk_level)}")
        print(f"  Interpretation : {r.interpretation}")
        print(f"  Recommendation : {r.recommendation}")
        print(f"\n  Parameter breakdown:")
        for k, v in r.breakdown.items():
            if isinstance(v, bool):
                marker = "✓" if v else "—"
                print(f"    {marker}  {k}")
            else:
                print(f"    {v:+.0f}  {k}")
        print()

    if result.reasoning:
        print(f"{'─' * 60}")
        print(f"{BOLD}AI Reasoning:{RESET}")
        print(result.reasoning)

    print(f"\n{'─' * 60}")
    print(f"Model: {result.model_used}  |  Calculators invoked: {len(result.tool_calls_made)}")


if __name__ == "__main__":
    main()
