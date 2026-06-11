 # Clinical Calculator Tool-Use System

An LLM-powered clinical decision support system that:
1. Accepts **free-text clinical notes** as input
2. Uses **Claude with tool-use** to extract relevant findings
3. Automatically invokes the appropriate **clinical calculator(s)**
4. Returns the **score, risk level, interpretation, and recommendation**

## Supported Calculators

| Calculator | Use case |
|---|---|
| `wells_dvt` | Deep Vein Thrombosis pre-test probability |
| `wells_pe` | Pulmonary Embolism pre-test probability |
| `heart_score` | Chest pain / MACE risk stratification |
| `curb65` | Community-acquired pneumonia severity |
| `cha2ds2_vasc` | Stroke risk in atrial fibrillation |
| `chads2` | Stroke risk in AF (older tool) |
| `glasgow_coma_scale` | Level of consciousness |
| `psi_port` | Pneumonia Severity Index |
| `mews` | Modified Early Warning Score |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the FastAPI server

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the web UI, or http://localhost:8000/docs for the API docs.

### 4. CLI usage

```bash
python scripts/infer.py "72yo female, HR 118, haemoptysis, returned from long-haul flight 5 days ago, known breast cancer"

# From a file
python scripts/infer.py --file patient_note.txt

# JSON output
python scripts/infer.py --json "..."
```

## API Usage

### POST /analyse — LLM-powered analysis

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"clinical_note": "72yo female with pleuritic chest pain, HR 118, recent long-haul flight..."}'
```

**Response:**
```json
{
  "calculator_results": [
    {
      "calculator": "Wells Criteria for PE",
      "score": 9.0,
      "interpretation": "High probability of PE (>6 points)",
      "risk_level": "high",
      "recommendation": "CT pulmonary angiography (CTPA) indicated...",
      "breakdown": { "Heart rate >100 bpm": 1.5, ... }
    }
  ],
  "reasoning": "The patient presents with ...",
  "extracted_parameters": { "wells_pe": { "heart_rate_gt100": true, ... } },
  "model_used": "claude-opus-4-5",
  "tool_calls_made": [...]
}
```

### POST /calculate — Direct calculator invocation

```bash
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"calculator": "curb65", "parameters": {"confusion": true, "urea_gt7": true, "age_gte65": true}}'
```

### GET /calculators — List available calculators

```bash
curl http://localhost:8000/calculators
```

## Training Data Generation

```bash
python training/generate_data.py
# Outputs: training/data/train.jsonl, training/data/val.jsonl
```

The JSONL files contain multi-turn Anthropic messages-format examples suitable for fine-tuning.

## Evaluation

```bash
python evaluation/evaluate.py
```

Evaluates the system on built-in ground-truth cases and reports:
- Calculator selection accuracy
- Score accuracy
- Risk-level accuracy
- Parameter extraction accuracy

## Project Structure

```
clinical-calculator/
├── app/
│   ├── main.py               # FastAPI application
│   ├── inference.py          # LLM tool-use pipeline
│   ├── calculators/
│   │   ├── calculators.py    # All clinical calculator implementations
│   │   └── tools.py          # Anthropic tool definitions
│   └── templates/
│       └── index.html        # Web UI
├── training/
│   └── generate_data.py      # Training data generator
├── evaluation/
│   └── evaluate.py           # Evaluation framework
├── scripts/
│   └── infer.py              # CLI tool
├── requirements.txt
└── README.md
```

## Architecture

```
Clinical Note (free text)
        │
        ▼
  Claude API (with tool definitions)
        │
        ├── Extracts findings
        ├── Selects calculator(s)
        └── Calls tool with structured parameters
              │
              ▼
        Python Calculator Functions
        (deterministic, evidence-based)
              │
              ▼
        Score + Interpretation + Recommendation
              │
              ▼
        Claude provides reasoning summary
```

## Clinical Disclaimer

This system is intended for **educational and research purposes only**. Clinical calculators implemented here are based on published scoring systems but should **not** be used as the sole basis for clinical decision-making. Always verify scores manually and use clinical judgement.

## Citation

Original repository: [lucidrains/clinical-calculator-tooluse](https://github.com/lucidrains/clinical-calculator-tooluse)

Built on the Toolformer paradigm (Schick et al., 2023) applied to clinical decision support.
