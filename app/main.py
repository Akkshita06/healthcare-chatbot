"""
FastAPI backend for the Clinical Calculator Tool-Use system.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.calculators.calculators import CALCULATOR_REGISTRY, CalculatorResult
from app.inference import InferenceResult, run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Calculator Tool-Use API",
    description=(
        "LLM-powered clinical decision support: accepts free-text clinical notes, "
        "extracts findings, and automatically applies appropriate clinical calculators."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    clinical_note: str = Field(
        ...,
        description="Free-text clinical note, patient history, or presenting complaint",
        min_length=10,
        examples=["64yo male with 3-day history of pleuritic chest pain, haemoptysis, "
                  "HR 112 bpm, recent long-haul flight 1 week ago. Known DVT 2 years ago."],
    )
    model: str = Field(
        default="claude-opus-4-5",
        description="Anthropic model to use for extraction and reasoning",
    )


class CalculatorResultOut(BaseModel):
    calculator: str
    score: float
    interpretation: str
    risk_level: str
    recommendation: str
    breakdown: dict[str, Any]


class AnalyseResponse(BaseModel):
    calculator_results: list[CalculatorResultOut]
    reasoning: str
    extracted_parameters: dict[str, dict[str, Any]]
    model_used: str
    tool_calls_made: list[dict]


class DirectCalculateRequest(BaseModel):
    calculator: str = Field(..., description="Calculator name (e.g. 'wells_pe', 'curb65')")
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    try:
        with open("app/templates/index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h2>UI not found. See /docs for API.</h2>")


@app.post("/analyse", response_model=AnalyseResponse, tags=["LLM"])
async def analyse_clinical_note(request: AnalyseRequest):
    """
    Accept a free-text clinical note and use an LLM to extract findings,
    select the appropriate calculator(s), and return scored results.
    """
    backend = os.environ.get("INFERENCE_BACKEND", "anthropic").lower()
    key_name = "GROQ_API_KEY" if backend == "groq" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{key_name} environment variable not set.")

    try:
        result: InferenceResult = run_inference(
            clinical_note=request.clinical_note,
            api_key=api_key,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyseResponse(
        calculator_results=[
            CalculatorResultOut(**r.to_dict()) for r in result.calculator_results
        ],
        reasoning=result.reasoning,
        extracted_parameters=result.extracted_parameters,
        model_used=result.model_used,
        tool_calls_made=result.tool_calls_made,
    )


@app.post("/calculate", response_model=CalculatorResultOut, tags=["Calculators"])
async def direct_calculate(request: DirectCalculateRequest):
    """
    Directly invoke a specific calculator with structured parameters
    (no LLM intermediary).
    """
    if request.calculator not in CALCULATOR_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown calculator '{request.calculator}'. "
                   f"Available: {list(CALCULATOR_REGISTRY.keys())}",
        )
    try:
        result: CalculatorResult = CALCULATOR_REGISTRY[request.calculator](
            **request.parameters
        )
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CalculatorResultOut(**result.to_dict())


@app.get("/calculators", tags=["Calculators"])
async def list_calculators():
    """List all available clinical calculators."""
    return {"calculators": list(CALCULATOR_REGISTRY.keys())}


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok"}