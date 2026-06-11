"""
Anthropic tool definitions for every clinical calculator.
These are passed to the Claude API so the model can invoke calculators
as structured tool calls when processing free-text clinical notes.
"""

from __future__ import annotations

CLINICAL_CALCULATOR_TOOLS: list[dict] = [
    {
        "name": "wells_dvt",
        "description": (
            "Calculate the Wells Criteria score for Deep Vein Thrombosis (DVT). "
            "Use when a patient presents with leg swelling, pain, tenderness, or when "
            "DVT is in the differential diagnosis. Helps stratify pre-test probability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "active_cancer": {
                    "type": "boolean",
                    "description": "Active cancer (treatment within 6 months, or palliative)"
                },
                "paralysis_or_plaster": {
                    "type": "boolean",
                    "description": "Paralysis, paresis, or recent plaster immobilisation of lower extremity"
                },
                "bedridden_gt3_days_or_major_surgery": {
                    "type": "boolean",
                    "description": "Bedridden >3 days or major surgery <12 weeks requiring general or regional anaesthesia"
                },
                "localized_tenderness": {
                    "type": "boolean",
                    "description": "Localised tenderness along the deep venous system"
                },
                "entire_leg_swollen": {
                    "type": "boolean",
                    "description": "Entire leg swollen"
                },
                "calf_swelling_gt3cm": {
                    "type": "boolean",
                    "description": "Calf swelling >3 cm compared to asymptomatic leg (10 cm below tibial tuberosity)"
                },
                "pitting_oedema": {
                    "type": "boolean",
                    "description": "Pitting oedema confined to symptomatic leg"
                },
                "collateral_superficial_veins": {
                    "type": "boolean",
                    "description": "Collateral superficial veins (non-varicose)"
                },
                "alternative_diagnosis_as_likely": {
                    "type": "boolean",
                    "description": "Alternative diagnosis at least as likely as DVT (subtracts 2 points)"
                },
                "previous_dvt": {
                    "type": "boolean",
                    "description": "Previously documented DVT"
                },
            },
            "required": [],
        },
    },
    {
        "name": "wells_pe",
        "description": (
            "Calculate the Wells Criteria score for Pulmonary Embolism (PE). "
            "Use when a patient presents with dyspnoea, pleuritic chest pain, haemoptysis, "
            "tachycardia, or when PE is in the differential diagnosis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "clinical_signs_dvt": {
                    "type": "boolean",
                    "description": "Clinical signs and symptoms of DVT (minimum of leg swelling and pain with palpation of the deep veins)"
                },
                "pe_most_likely_diagnosis": {
                    "type": "boolean",
                    "description": "PE is the #1 diagnosis, or equally likely as an alternative diagnosis"
                },
                "heart_rate_gt100": {
                    "type": "boolean",
                    "description": "Heart rate >100 bpm"
                },
                "immobilisation_or_surgery": {
                    "type": "boolean",
                    "description": "Immobilisation ≥3 consecutive days OR surgery in the previous 4 weeks"
                },
                "previous_pe_or_dvt": {
                    "type": "boolean",
                    "description": "Previous objectively diagnosed PE or DVT"
                },
                "haemoptysis": {
                    "type": "boolean",
                    "description": "Haemoptysis (coughing up blood)"
                },
                "malignancy": {
                    "type": "boolean",
                    "description": "Malignancy with treatment in past 6 months or palliative"
                },
            },
            "required": [],
        },
    },
    {
        "name": "heart_score",
        "description": (
            "Calculate the HEART Score for Major Adverse Cardiac Events (MACE). "
            "Use for patients presenting with chest pain to risk-stratify for 6-week MACE. "
            "Components: History, ECG, Age, Risk factors, Troponin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "history": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "History: 0=slightly suspicious (non-specific), 1=moderately suspicious, 2=highly suspicious for ACS"
                },
                "ecg": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "ECG: 0=normal, 1=non-specific repolarisation disturbance (LBBB, LVH, early repolarisation, digoxin changes), 2=significant ST deviation"
                },
                "age": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "Age: 0=<45 years, 1=45–64 years, 2=≥65 years"
                },
                "risk_factors": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "Risk factors: 0=no known risk factors, 1=1–2 risk factors (HTN, hypercholesterolaemia, DM, obesity, smoking, family hx), 2=≥3 risk factors or history of atherosclerotic disease"
                },
                "troponin": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "Troponin: 0=≤normal limit, 1=1–3× normal limit, 2=>3× normal limit"
                },
            },
            "required": [],
        },
    },
    {
        "name": "curb65",
        "description": (
            "Calculate the CURB-65 score for Community-Acquired Pneumonia (CAP) severity. "
            "Use when a patient is diagnosed with or suspected of having CAP to determine "
            "appropriate level of care (outpatient vs inpatient vs ICU)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confusion": {
                    "type": "boolean",
                    "description": "New-onset confusion (AMTS ≤8 or new disorientation to person, place, or time)"
                },
                "urea_gt7": {
                    "type": "boolean",
                    "description": "Blood urea nitrogen >7 mmol/L (or BUN >19 mg/dL)"
                },
                "respiratory_rate_gte30": {
                    "type": "boolean",
                    "description": "Respiratory rate ≥30 breaths/minute"
                },
                "low_bp": {
                    "type": "boolean",
                    "description": "Low blood pressure: systolic <90 mmHg or diastolic ≤60 mmHg"
                },
                "age_gte65": {
                    "type": "boolean",
                    "description": "Age ≥65 years"
                },
            },
            "required": [],
        },
    },
    {
        "name": "cha2ds2_vasc",
        "description": (
            "Calculate the CHA₂DS₂-VASc score for stroke risk in Atrial Fibrillation (AF). "
            "Use when a patient has AF to determine appropriateness of anticoagulation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chf": {
                    "type": "boolean",
                    "description": "Congestive heart failure (or LVEF <40%)"
                },
                "hypertension": {
                    "type": "boolean",
                    "description": "Hypertension (resting BP >140/90 on at least 2 occasions or current antihypertensive treatment)"
                },
                "age_gte75": {
                    "type": "boolean",
                    "description": "Age ≥75 years (scores 2 points)"
                },
                "diabetes": {
                    "type": "boolean",
                    "description": "Diabetes mellitus (fasting glucose >125 mg/dL or on diabetes treatment)"
                },
                "stroke_or_tia": {
                    "type": "boolean",
                    "description": "Prior stroke, TIA, or thromboembolism (scores 2 points)"
                },
                "vascular_disease": {
                    "type": "boolean",
                    "description": "Vascular disease (prior MI, peripheral artery disease, or aortic plaque)"
                },
                "age_65_74": {
                    "type": "boolean",
                    "description": "Age 65–74 years"
                },
                "female_sex": {
                    "type": "boolean",
                    "description": "Female sex category"
                },
            },
            "required": [],
        },
    },
    {
        "name": "glasgow_coma_scale",
        "description": (
            "Calculate the Glasgow Coma Scale (GCS) for level of consciousness. "
            "Use in patients with altered consciousness, head trauma, or neurological emergency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "eye_opening": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4],
                    "description": "Eye opening: 1=none, 2=to pain, 3=to voice, 4=spontaneous"
                },
                "verbal_response": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "Verbal response: 1=none, 2=incomprehensible sounds, 3=inappropriate words, 4=confused, 5=oriented"
                },
                "motor_response": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5, 6],
                    "description": "Motor response: 1=none, 2=extension to pain, 3=abnormal flexion, 4=withdrawal, 5=localises pain, 6=obeys commands"
                },
            },
            "required": [],
        },
    },
    {
        "name": "curb65",
        "description": (
            "Calculate the CURB-65 score for Community-Acquired Pneumonia severity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confusion": {"type": "boolean", "description": "New confusion"},
                "urea_gt7": {"type": "boolean", "description": "BUN >7 mmol/L"},
                "respiratory_rate_gte30": {"type": "boolean", "description": "RR ≥30/min"},
                "low_bp": {"type": "boolean", "description": "SBP <90 or DBP ≤60"},
                "age_gte65": {"type": "boolean", "description": "Age ≥65 years"},
            },
            "required": [],
        },
    },
    {
        "name": "mews",
        "description": (
            "Calculate the Modified Early Warning Score (MEWS) for acutely ill inpatients. "
            "Use to identify patients at risk of clinical deterioration who may need urgent review or escalation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sbp": {
                    "type": "integer",
                    "description": "Systolic blood pressure in mmHg"
                },
                "heart_rate": {
                    "type": "integer",
                    "description": "Heart rate in beats per minute"
                },
                "respiratory_rate": {
                    "type": "integer",
                    "description": "Respiratory rate in breaths per minute"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius"
                },
                "avpu": {
                    "type": "string",
                    "enum": ["A", "V", "P", "U"],
                    "description": "AVPU consciousness scale: A=Alert, V=responds to Voice, P=responds to Pain, U=Unresponsive"
                },
            },
            "required": [],
        },
    },
]

# De-duplicate by name (we accidentally had curb65 twice above)
seen: set[str] = set()
CLINICAL_CALCULATOR_TOOLS_DEDUPED: list[dict] = []
for t in CLINICAL_CALCULATOR_TOOLS:
    if t["name"] not in seen:
        seen.add(t["name"])
        CLINICAL_CALCULATOR_TOOLS_DEDUPED.append(t)

CLINICAL_CALCULATOR_TOOLS = CLINICAL_CALCULATOR_TOOLS_DEDUPED
