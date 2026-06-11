"""
LLM-powered clinical calculator inference engine.

Supports two backends, selected via the INFERENCE_BACKEND env var:
  - "anthropic" (default) — uses Claude with native tool-use
  - "groq"                — uses Groq via JSON-mode structured extraction

Flow (both backends):
  1. Accept free-text clinical note
  2. LLM extracts findings and selects which calculator(s) to invoke
  3. We execute the real Python calculator with the extracted params
  4. Return score, interpretation, reasoning, and extracted findings
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from app.calculators.calculators import CALCULATOR_REGISTRY, CalculatorResult
from app.calculators.tools import CLINICAL_CALCULATOR_TOOLS

logger = logging.getLogger(__name__)

# ── Shared system prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical decision support assistant specialising in evidence-based risk scoring tools.

Your task:
1. Read the free-text clinical note carefully.
2. Identify which clinical calculator(s) are most appropriate for this presentation.
3. Extract every relevant parameter from the note.
4. Call the appropriate calculator tool(s) with the extracted parameters.
5. Only call a calculator if there is sufficient clinical information to justify it.

IMPORTANT RULES:
- Only call calculators that are clinically relevant to the presentation.
- If a parameter is not mentioned, do NOT assume a value — leave it as false/default.
- If the note is clearly insufficient, say so and ask what additional information is needed.
- You may call more than one calculator if multiple are applicable.
- After the tool calls, provide a brief clinical reasoning summary.
"""

# Groq JSON-mode prompt: asks the model to return structured extraction instead
# of making native tool calls (Groq models don't support Anthropic-style tools).
GROQ_SYSTEM_PROMPT = """You are a clinical decision support assistant.

Analyse the clinical note and return ONLY a JSON object — no markdown, no explanation outside the JSON.

Available calculators: wells_dvt, wells_pe, heart_score, curb65, cha2ds2_vasc, chads2, glasgow_coma_scale, psi_port, mews

Return this exact structure:
{
  "calculators": [
    {
      "name": "<calculator_name>",
      "parameters": { <param>: <value>, ... },
      "reasoning": "<brief explanation of why this calculator was chosen and how params were extracted>"
    }
  ],
  "overall_reasoning": "<summary of clinical presentation>"
}

Parameter reference:
- wells_dvt: active_cancer, paralysis_or_plaster, bedridden_gt3_days_or_major_surgery, localized_tenderness, entire_leg_swollen, calf_swelling_gt3cm, pitting_oedema, collateral_superficial_veins, alternative_diagnosis_as_likely, previous_dvt (all boolean)
- wells_pe: clinical_signs_dvt, pe_most_likely_diagnosis, heart_rate_gt100, immobilisation_or_surgery, previous_pe_or_dvt, haemoptysis, malignancy (all boolean)
- heart_score: history (0-2), ecg (0-2), age (0-2), risk_factors (0-2), troponin (0-2)
- curb65: confusion, urea_gt7, respiratory_rate_gte30, low_bp, age_gte65 (all boolean)
- cha2ds2_vasc: chf, hypertension, age_gte75, diabetes, stroke_or_tia, vascular_disease, age_65_74, female_sex (all boolean)
- glasgow_coma_scale: eye_opening (1-4), verbal_response (1-5), motor_response (1-6)
- mews: sbp (int mmHg), heart_rate (int bpm), respiratory_rate (int /min), temperature (float °C), avpu ("A"/"V"/"P"/"U")

Only include calculators relevant to the presentation. Omit unknown parameters (they'll use defaults).
"""


@dataclass
class InferenceResult:
    clinical_note: str
    calculator_results: list[CalculatorResult]
    reasoning: str
    extracted_parameters: dict[str, dict[str, Any]]
    model_used: str
    tool_calls_made: list[dict]


# ── Anthropic backend ────────────────────────────────────────────────────────

def _run_anthropic(
    clinical_note: str,
    api_key: str,
    model: str,
) -> InferenceResult:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = [
        {"role": "user", "content": f"Please analyse the following clinical note and apply the appropriate clinical calculator(s):\n\n{clinical_note}"}
    ]

    calculator_results: list[CalculatorResult] = []
    extracted_parameters: dict[str, dict[str, Any]] = {}
    tool_calls_made: list[dict] = []
    reasoning: str = ""

    for _round in range(5):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=CLINICAL_CALCULATOR_TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                reasoning = block.text.strip()

        if response.stop_reason == "end_turn":
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input
            tool_calls_made.append({"tool": tool_name, "input": tool_input})
            extracted_parameters[tool_name] = tool_input

            if tool_name in CALCULATOR_REGISTRY:
                try:
                    result = CALCULATOR_REGISTRY[tool_name](**tool_input)
                    calculator_results.append(result)
                    result_content = json.dumps(result.to_dict())
                except Exception as exc:
                    logger.exception("Calculator %s failed", tool_name)
                    result_content = json.dumps({"error": str(exc)})
            else:
                result_content = json.dumps({"error": f"Unknown calculator: {tool_name}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        messages.append({"role": "user", "content": tool_results})

    return InferenceResult(
        clinical_note=clinical_note,
        calculator_results=calculator_results,
        reasoning=reasoning,
        extracted_parameters=extracted_parameters,
        model_used=model,
        tool_calls_made=tool_calls_made,
    )


# ── Groq backend ─────────────────────────────────────────────────────────────

def _run_groq(
    clinical_note: str,
    api_key: str,
    model: str,
) -> InferenceResult:
    from groq import Groq

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user", "content": clinical_note},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Groq returned invalid JSON: {exc}\n\nRaw: {raw}") from exc

    calculator_results: list[CalculatorResult] = []
    extracted_parameters: dict[str, dict[str, Any]] = {}
    tool_calls_made: list[dict] = []

    for entry in parsed.get("calculators", []):
        name = entry.get("name", "")
        params = entry.get("parameters", {})

        tool_calls_made.append({"tool": name, "input": params})
        extracted_parameters[name] = params

        if name in CALCULATOR_REGISTRY:
            try:
                result = CALCULATOR_REGISTRY[name](**params)
                calculator_results.append(result)
            except Exception as exc:
                logger.exception("Calculator %s failed", name)
        else:
            logger.warning("Groq returned unknown calculator: %s", name)

    reasoning = parsed.get("overall_reasoning", "")

    return InferenceResult(
        clinical_note=clinical_note,
        calculator_results=calculator_results,
        reasoning=reasoning,
        extracted_parameters=extracted_parameters,
        model_used=model,
        tool_calls_made=tool_calls_made,
    )


# ── Public entry point ───────────────────────────────────────────────────────

def run_inference(
    clinical_note: str,
    api_key: str | None = None,
    model: str | None = None,
) -> InferenceResult:
    """
    Run the LLM → calculator pipeline.

    Backend is selected via INFERENCE_BACKEND env var ("anthropic" or "groq").
    API key and model fall back to env vars if not provided:
      - Anthropic: ANTHROPIC_API_KEY, default model claude-opus-4-5
      - Groq:      GROQ_API_KEY,      default model llama-3.3-70b-versatile
    """
    backend = os.environ.get("INFERENCE_BACKEND", "anthropic").lower()

    if backend == "groq":
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        resolved_model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return _run_groq(clinical_note, resolved_key, resolved_model)
    else:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5")
        return _run_anthropic(clinical_note, resolved_key, resolved_model)