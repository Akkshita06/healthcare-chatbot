"""
Training data generation for LLM fine-tuning on clinical calculator tool use.

Generates JSONL files in Anthropic's messages format, suitable for:
  - Supervised fine-tuning (SFT) on tool-use examples
  - Evaluation benchmarking

Each example contains:
  - system: the clinical assistant system prompt
  - messages: [user clinical note → assistant tool call → tool result → assistant explanation]
  - metadata: calculator name, expected score, risk level
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Seed for reproducibility
random.seed(42)


@dataclass
class TrainingExample:
    system: str
    messages: list[dict]
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Clinical vignette templates ──────────────────────────────────────────────

WELLS_PE_VIGNETTES = [
    # (params_dict, expected_score, clinical_note_template)
    {
        "params": {
            "clinical_signs_dvt": True,
            "pe_most_likely_diagnosis": True,
            "heart_rate_gt100": True,
            "immobilisation_or_surgery": True,
            "previous_pe_or_dvt": True,
            "haemoptysis": True,
            "malignancy": True,
        },
        "expected_score": 12.5,
        "note": (
            "78-year-old woman presenting with acute dyspnoea and haemoptysis. "
            "Her heart rate is 118 bpm. She had hip surgery 3 weeks ago and was "
            "immobilised for a week. She has a history of DVT treated 2 years ago "
            "and is currently on palliative chemotherapy for metastatic breast cancer. "
            "Left leg is swollen with tenderness over the popliteal fossa. "
            "PE is the most likely diagnosis."
        ),
    },
    {
        "params": {
            "heart_rate_gt100": True,
            "immobilisation_or_surgery": False,
            "previous_pe_or_dvt": False,
            "haemoptysis": False,
            "malignancy": False,
            "clinical_signs_dvt": False,
            "pe_most_likely_diagnosis": False,
        },
        "expected_score": 1.5,
        "note": (
            "35-year-old male with 2-hour history of mild dyspnoea and heart rate of 108 bpm. "
            "No leg swelling. No prior DVT or PE. No recent surgery or immobility. "
            "No haemoptysis. No malignancy. Anxiety disorder in background. "
            "Alternative diagnoses such as panic attack considered equally likely."
        ),
    },
    {
        "params": {
            "clinical_signs_dvt": True,
            "pe_most_likely_diagnosis": True,
            "heart_rate_gt100": False,
            "immobilisation_or_surgery": False,
            "previous_pe_or_dvt": False,
            "haemoptysis": False,
            "malignancy": False,
        },
        "expected_score": 6.0,
        "note": (
            "55-year-old female with sudden-onset right-sided pleuritic chest pain. "
            "Examination reveals right calf swelling and tenderness. "
            "HR is 90 bpm. No recent surgery or immobilisation. "
            "No history of DVT or PE. No haemoptysis. No malignancy. "
            "PE is the most likely diagnosis."
        ),
    },
]

CURB65_VIGNETTES = [
    {
        "params": {
            "confusion": True,
            "urea_gt7": True,
            "respiratory_rate_gte30": True,
            "low_bp": True,
            "age_gte65": True,
        },
        "expected_score": 5,
        "note": (
            "84-year-old female presenting with 2-day history of productive cough and fever. "
            "She is confused (AMTS 6/10), blood pressure 82/54 mmHg, respiratory rate 34/min, "
            "urea 11 mmol/L. She has consolidation on CXR consistent with CAP."
        ),
    },
    {
        "params": {
            "confusion": False,
            "urea_gt7": False,
            "respiratory_rate_gte30": False,
            "low_bp": False,
            "age_gte65": False,
        },
        "expected_score": 0,
        "note": (
            "28-year-old male with 3-day history of cough, fever 38.2°C, and mild dyspnoea. "
            "Alert and oriented. RR 18/min, BP 122/76 mmHg. Urea 4.2 mmol/L. "
            "CXR shows left lower lobe infiltrate consistent with CAP."
        ),
    },
    {
        "params": {
            "confusion": False,
            "urea_gt7": True,
            "respiratory_rate_gte30": True,
            "low_bp": False,
            "age_gte65": True,
        },
        "expected_score": 3,
        "note": (
            "71-year-old male with CAP. He is alert, BP 130/80. RR 31/min. "
            "Urea elevated at 9.2 mmol/L. CXR shows bilateral infiltrates."
        ),
    },
]

HEART_SCORE_VIGNETTES = [
    {
        "params": {
            "history": 2,
            "ecg": 2,
            "age": 2,
            "risk_factors": 2,
            "troponin": 2,
        },
        "expected_score": 10,
        "note": (
            "72-year-old male with crushing central chest pain radiating to the left arm "
            "for 2 hours. History is highly suspicious for ACS. Known hypertensive, diabetic, "
            "ex-smoker, with prior MI. ECG shows 3 mm ST depression in V4-V6. "
            "Troponin is 12× upper limit of normal."
        ),
    },
    {
        "params": {
            "history": 0,
            "ecg": 0,
            "age": 0,
            "risk_factors": 0,
            "troponin": 0,
        },
        "expected_score": 0,
        "note": (
            "19-year-old female with sharp, positional chest pain for 1 hour, "
            "worse on deep breathing, improved when leaning forward. "
            "ECG is normal. Troponin is within normal limits. "
            "No cardiovascular risk factors. Presentation is mildly suspicious — "
            "pleuritic / musculoskeletal aetiology more likely."
        ),
    },
]

CHA2DS2_VIGNETTES = [
    {
        "params": {
            "chf": True,
            "hypertension": True,
            "age_gte75": True,
            "diabetes": True,
            "stroke_or_tia": True,
            "vascular_disease": False,
            "age_65_74": False,
            "female_sex": False,
        },
        "expected_score": 8,
        "note": (
            "78-year-old male with permanent AF. History of congestive heart failure, "
            "hypertension on treatment, type 2 diabetes, and a TIA 18 months ago. "
            "No vascular disease documented."
        ),
    },
    {
        "params": {
            "chf": False,
            "hypertension": False,
            "age_gte75": False,
            "diabetes": False,
            "stroke_or_tia": False,
            "vascular_disease": False,
            "age_65_74": False,
            "female_sex": False,
        },
        "expected_score": 0,
        "note": (
            "48-year-old male with newly diagnosed paroxysmal AF. "
            "No hypertension, no diabetes, no heart failure, no prior stroke or TIA. "
            "No vascular disease. Young healthy male with lone AF."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are a clinical decision support assistant specialising in evidence-based "
    "risk scoring tools. Analyse clinical notes, extract relevant parameters, and "
    "apply appropriate clinical calculators using the available tools."
)


def build_tool_use_example(
    vignette: dict,
    tool_name: str,
    tool_id: str = "toolu_01",
) -> TrainingExample:
    """Build a complete multi-turn training example from a vignette."""

    # Simulate what an ideal assistant response would look like
    tool_call_block = {
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": vignette["params"],
    }

    # Simulate tool result
    # We import and run the real calculator to get the actual result
    try:
        from app.calculators.calculators import CALCULATOR_REGISTRY
        result = CALCULATOR_REGISTRY[tool_name](**vignette["params"])
        result_content = json.dumps(result.to_dict())
    except Exception:
        result_content = json.dumps({"error": "calculator unavailable"})

    messages = [
        {
            "role": "user",
            "content": (
                "Please analyse the following clinical note and apply the appropriate "
                f"clinical calculator(s):\n\n{vignette['note']}"
            ),
        },
        {
            "role": "assistant",
            "content": [tool_call_block],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_content,
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                f"Based on the clinical note, I applied the {tool_name.replace('_', ' ').title()} "
                f"calculator. The key findings extracted were: "
                + ", ".join(
                    f"{k.replace('_', ' ')} = {v}"
                    for k, v in vignette["params"].items()
                    if v is not False and v != 0
                )
                + f". The score is {vignette['expected_score']}."
            ),
        },
    ]

    return TrainingExample(
        system=SYSTEM_PROMPT,
        messages=messages,
        metadata={
            "calculator": tool_name,
            "expected_score": vignette["expected_score"],
            "split": "train",
        },
    )


def generate_dataset(output_dir: str = "training/data") -> tuple[int, int]:
    """
    Generate train/validation JSONL datasets.

    Returns:
        (train_count, val_count)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_examples: list[TrainingExample] = []

    for v in WELLS_PE_VIGNETTES:
        all_examples.append(build_tool_use_example(v, "wells_pe"))

    for v in CURB65_VIGNETTES:
        all_examples.append(build_tool_use_example(v, "curb65"))

    for v in HEART_SCORE_VIGNETTES:
        all_examples.append(build_tool_use_example(v, "heart_score"))

    for v in CHA2DS2_VIGNETTES:
        all_examples.append(build_tool_use_example(v, "cha2ds2_vasc"))

    random.shuffle(all_examples)

    # 80/20 split
    split = max(1, int(len(all_examples) * 0.8))
    train_examples = all_examples[:split]
    val_examples = all_examples[split:]

    def write_jsonl(path: str, examples: list[TrainingExample]) -> None:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    write_jsonl(f"{output_dir}/train.jsonl", train_examples)
    write_jsonl(f"{output_dir}/val.jsonl", val_examples)

    print(f"✓ Generated {len(train_examples)} training examples → {output_dir}/train.jsonl")
    print(f"✓ Generated {len(val_examples)} validation examples  → {output_dir}/val.jsonl")
    return len(train_examples), len(val_examples)


if __name__ == "__main__":
    generate_dataset()
