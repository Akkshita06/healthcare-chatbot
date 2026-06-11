"""
Evaluation framework for the clinical calculator tool-use system.

Metrics:
  - Calculator selection accuracy (did the model choose the right calculator?)
  - Parameter extraction accuracy (were the right boolean/numeric params extracted?)
  - Score accuracy (did the final numeric score match ground truth?)
  - Risk-level accuracy (low/moderate/high/very_high)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.inference import run_inference


@dataclass
class EvalCase:
    note: str
    expected_calculator: str
    expected_score: int | float
    expected_risk_level: str
    expected_params: dict[str, Any] = field(default_factory=dict)


# Ground-truth evaluation cases
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        note=(
            "58-year-old man, alert and oriented, chest pain highly suspicious for ACS. "
            "ECG shows significant ST changes. Age category 45-64. "
            "3 cardiovascular risk factors (HTN, DM, smoking). Troponin 4× ULN."
        ),
        expected_calculator="heart_score",
        expected_score=8,
        expected_risk_level="high",
        expected_params={"history": 2, "ecg": 2, "age": 1, "risk_factors": 2, "troponin": 1},
    ),
    EvalCase(
        note=(
            "75-year-old woman with productive cough, fever 39°C, RR 32/min, "
            "BP 84/50, confused (disoriented to time and place), "
            "urea 8.5 mmol/L, CXR consolidation. CAP confirmed."
        ),
        expected_calculator="curb65",
        expected_score=5,
        expected_risk_level="high",
        expected_params={
            "confusion": True,
            "urea_gt7": True,
            "respiratory_rate_gte30": True,
            "low_bp": True,
            "age_gte65": True,
        },
    ),
    EvalCase(
        note=(
            "60-year-old female with AF, hypertension, diabetes, prior stroke 1 year ago, "
            "and documented heart failure with LVEF 35%. No vascular disease."
        ),
        expected_calculator="cha2ds2_vasc",
        expected_score=7,
        expected_risk_level="high",
        expected_params={
            "chf": True,
            "hypertension": True,
            "age_gte75": False,
            "diabetes": True,
            "stroke_or_tia": True,
            "vascular_disease": False,
            "age_65_74": True,
            "female_sex": True,
        },
    ),
    EvalCase(
        note=(
            "Acutely dyspnoeic 68-year-old with suspected PE. "
            "HR 120, calf DVT signs present. PE is the primary working diagnosis. "
            "Haemoptysis present. Known lung cancer on palliative treatment. "
            "Had abdominal surgery 2 weeks ago."
        ),
        expected_calculator="wells_pe",
        expected_score=11.0,
        expected_risk_level="high",
        expected_params={
            "clinical_signs_dvt": True,
            "pe_most_likely_diagnosis": True,
            "heart_rate_gt100": True,
            "immobilisation_or_surgery": True,
            "previous_pe_or_dvt": False,
            "haemoptysis": True,
            "malignancy": True,
        },
    ),
]


@dataclass
class EvalResult:
    case_index: int
    note: str
    expected_calculator: str
    expected_score: float
    expected_risk: str
    calculator_selected: str | None
    actual_score: float | None
    actual_risk: str | None
    calculator_correct: bool
    score_exact: bool
    risk_correct: bool
    param_accuracy: float


def evaluate(
    cases: list[EvalCase] | None = None,
    api_key: str | None = None,
) -> list[EvalResult]:
    cases = cases or EVAL_CASES
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    results: list[EvalResult] = []

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Running case: {case.expected_calculator}...")
        try:
            inference = run_inference(case.note, api_key=api_key)
        except Exception as exc:
            print(f"  ✗ Inference failed: {exc}")
            results.append(EvalResult(
                case_index=i,
                note=case.note,
                expected_calculator=case.expected_calculator,
                expected_score=float(case.expected_score),
                expected_risk=case.expected_risk_level,
                calculator_selected=None,
                actual_score=None,
                actual_risk=None,
                calculator_correct=False,
                score_exact=False,
                risk_correct=False,
                param_accuracy=0.0,
            ))
            continue

        # Find the best matching result
        matching = [r for r in inference.calculator_results
                    if r.calculator.lower().replace(" ", "_") in case.expected_calculator
                    or case.expected_calculator in r.calculator.lower().replace(" ", "_")]

        if not matching and inference.calculator_results:
            # Fallback: first result
            matching = [inference.calculator_results[0]]

        if matching:
            result = matching[0]
            calc_correct = case.expected_calculator.lower() in result.calculator.lower().replace(" ", "_") or \
                           result.calculator.lower().replace(" ", "_") in case.expected_calculator.lower()
            score_exact = abs(float(result.score) - float(case.expected_score)) < 0.1
            risk_correct = result.risk_level == case.expected_risk_level

            # Parameter accuracy
            extracted = inference.extracted_parameters.get(case.expected_calculator, {})
            if case.expected_params and extracted:
                correct_params = sum(
                    1 for k, v in case.expected_params.items()
                    if extracted.get(k) == v
                )
                param_acc = correct_params / len(case.expected_params)
            else:
                param_acc = 0.0

            er = EvalResult(
                case_index=i,
                note=case.note[:80] + "...",
                expected_calculator=case.expected_calculator,
                expected_score=float(case.expected_score),
                expected_risk=case.expected_risk_level,
                calculator_selected=result.calculator,
                actual_score=float(result.score),
                actual_risk=result.risk_level,
                calculator_correct=calc_correct,
                score_exact=score_exact,
                risk_correct=risk_correct,
                param_accuracy=param_acc,
            )
        else:
            er = EvalResult(
                case_index=i,
                note=case.note[:80] + "...",
                expected_calculator=case.expected_calculator,
                expected_score=float(case.expected_score),
                expected_risk=case.expected_risk_level,
                calculator_selected=None,
                actual_score=None,
                actual_risk=None,
                calculator_correct=False,
                score_exact=False,
                risk_correct=False,
                param_accuracy=0.0,
            )

        status = "✓" if er.calculator_correct and er.score_exact else "✗"
        print(f"  {status} calculator={er.calculator_correct} score={er.score_exact} "
              f"risk={er.risk_correct} param_acc={er.param_accuracy:.1%}")
        results.append(er)

    return results


def print_summary(results: list[EvalResult]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return

    calc_acc = sum(r.calculator_correct for r in results) / n
    score_acc = sum(r.score_exact for r in results) / n
    risk_acc = sum(r.risk_correct for r in results) / n
    avg_param = sum(r.param_accuracy for r in results) / n

    print("\n" + "═" * 50)
    print("EVALUATION SUMMARY")
    print("═" * 50)
    print(f"  Cases evaluated      : {n}")
    print(f"  Calculator accuracy  : {calc_acc:.1%}")
    print(f"  Score accuracy       : {score_acc:.1%}")
    print(f"  Risk-level accuracy  : {risk_acc:.1%}")
    print(f"  Avg param accuracy   : {avg_param:.1%}")
    print("═" * 50)


if __name__ == "__main__":
    results = evaluate()
    print_summary(results)

    out_path = "evaluation/results.json"
    Path("evaluation").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"\nDetailed results → {out_path}")
