"""
Clinical calculators used by the LLM tool-use system.
Each calculator is implemented as a pure function that accepts structured
input and returns a score, interpretation, and clinical recommendation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class CalculatorResult:
    calculator: str
    score: int | float
    interpretation: str
    risk_level: str          # low / moderate / high / very_high
    recommendation: str
    breakdown: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Wells Criteria for DVT
# ---------------------------------------------------------------------------

def wells_dvt(
    active_cancer: bool = False,
    paralysis_or_plaster: bool = False,
    bedridden_gt3_days_or_major_surgery: bool = False,
    localized_tenderness: bool = False,
    entire_leg_swollen: bool = False,
    calf_swelling_gt3cm: bool = False,
    pitting_oedema: bool = False,
    collateral_superficial_veins: bool = False,
    alternative_diagnosis_as_likely: bool = False,
    previous_dvt: bool = False,
) -> CalculatorResult:
    """Wells Criteria for Deep Vein Thrombosis (DVT)."""
    breakdown: dict[str, int] = {
        "Active cancer": int(active_cancer),
        "Paralysis / plaster immobilisation": int(paralysis_or_plaster),
        "Bedridden >3 days or major surgery <12 weeks": int(bedridden_gt3_days_or_major_surgery),
        "Localised tenderness along deep venous system": int(localized_tenderness),
        "Entire leg swollen": int(entire_leg_swollen),
        "Calf swelling >3 cm vs asymptomatic side": int(calf_swelling_gt3cm),
        "Pitting oedema (symptomatic leg only)": int(pitting_oedema),
        "Collateral superficial veins": int(collateral_superficial_veins),
        "Alternative diagnosis at least as likely (-2)": -2 if alternative_diagnosis_as_likely else 0,
        "Previously documented DVT": int(previous_dvt),
    }
    score = sum(breakdown.values())

    if score >= 2:
        risk_level = "high"
        interpretation = "High probability of DVT (≥2 points)"
        recommendation = "DVT likely. Perform proximal leg ultrasound. If negative, repeat in 6–8 days or obtain D-dimer."
    elif score == 1:
        risk_level = "moderate"
        interpretation = "Moderate probability of DVT (1 point)"
        recommendation = "DVT possible. Obtain D-dimer. If elevated, proceed to ultrasound."
    else:
        risk_level = "low"
        interpretation = "Low probability of DVT (≤0 points)"
        recommendation = "DVT unlikely. Obtain D-dimer; if negative, DVT excluded without imaging."

    return CalculatorResult(
        calculator="Wells Criteria for DVT",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Wells Criteria for Pulmonary Embolism
# ---------------------------------------------------------------------------

def wells_pe(
    clinical_signs_dvt: bool = False,
    pe_most_likely_diagnosis: bool = False,
    heart_rate_gt100: bool = False,
    immobilisation_or_surgery: bool = False,
    previous_pe_or_dvt: bool = False,
    haemoptysis: bool = False,
    malignancy: bool = False,
) -> CalculatorResult:
    """Wells Criteria for Pulmonary Embolism (PE)."""
    breakdown: dict[str, float] = {
        "Clinical signs/symptoms of DVT": 3.0 if clinical_signs_dvt else 0.0,
        "PE is #1 diagnosis, or equally likely": 3.0 if pe_most_likely_diagnosis else 0.0,
        "Heart rate >100 bpm": 1.5 if heart_rate_gt100 else 0.0,
        "Immobilisation ≥3 days or surgery in past 4 weeks": 1.5 if immobilisation_or_surgery else 0.0,
        "Previous DVT/PE": 1.5 if previous_pe_or_dvt else 0.0,
        "Haemoptysis": 1.0 if haemoptysis else 0.0,
        "Malignancy with treatment in past 6 months or palliative": 1.0 if malignancy else 0.0,
    }
    score = sum(breakdown.values())

    if score > 6:
        risk_level = "high"
        interpretation = "High probability of PE (>6 points)"
        recommendation = "CT pulmonary angiography (CTPA) indicated. Consider empiric anticoagulation."
    elif score > 4:
        risk_level = "moderate"
        interpretation = "Moderate probability of PE (>4–6 points)"
        recommendation = "Obtain D-dimer; if elevated or positive, proceed to CTPA."
    else:
        risk_level = "low"
        interpretation = "Low probability of PE (≤4 points)"
        recommendation = "Obtain D-dimer; if negative with Wells score ≤4, PE can be excluded without imaging."

    return CalculatorResult(
        calculator="Wells Criteria for PE",
        score=round(score, 1),
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# HEART Score (chest pain)
# ---------------------------------------------------------------------------

def heart_score(
    history: int = 0,         # 0=slightly suspicious, 1=moderately suspicious, 2=highly suspicious
    ecg: int = 0,             # 0=normal, 1=non-specific repolarisation disturbance, 2=significant ST deviation
    age: int = 0,             # 0=<45, 1=45–64, 2=≥65
    risk_factors: int = 0,    # 0=none known, 1=1–2 risk factors, 2=≥3 or history of atherosclerosis
    troponin: int = 0,        # 0=≤normal, 1=1–3× normal, 2=>3× normal
) -> CalculatorResult:
    """HEART Score for Major Adverse Cardiac Events."""
    breakdown = {
        "History": history,
        "ECG": ecg,
        "Age": age,
        "Risk factors": risk_factors,
        "Troponin": troponin,
    }
    score = sum(breakdown.values())

    if score >= 7:
        risk_level = "high"
        interpretation = "High risk (7–10): ~65% risk of MACE"
        recommendation = "Early invasive strategy recommended. Cardiology consultation."
    elif score >= 4:
        risk_level = "moderate"
        interpretation = "Moderate risk (4–6): ~12–17% risk of MACE"
        recommendation = "Observation with serial ECGs and troponins. Consider stress testing or cardiology referral."
    else:
        risk_level = "low"
        interpretation = "Low risk (0–3): ~1–2% risk of MACE"
        recommendation = "Patient may be discharged with outpatient follow-up if symptoms resolve."

    return CalculatorResult(
        calculator="HEART Score",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# CURB-65 (community-acquired pneumonia severity)
# ---------------------------------------------------------------------------

def curb65(
    confusion: bool = False,
    urea_gt7: bool = False,
    respiratory_rate_gte30: bool = False,
    low_bp: bool = False,      # systolic <90 or diastolic ≤60
    age_gte65: bool = False,
) -> CalculatorResult:
    """CURB-65 Score for Community-Acquired Pneumonia Severity."""
    breakdown = {
        "Confusion (new disorientation)": int(confusion),
        "Urea >7 mmol/L (BUN >19 mg/dL)": int(urea_gt7),
        "Respiratory rate ≥30 /min": int(respiratory_rate_gte30),
        "Low BP (SBP <90 or DBP ≤60 mmHg)": int(low_bp),
        "Age ≥65 years": int(age_gte65),
    }
    score = sum(breakdown.values())

    if score >= 3:
        risk_level = "high"
        interpretation = f"Severe CAP (score {score}/5): 30-day mortality ~14–40%"
        recommendation = "Hospitalise; consider ICU admission for score ≥4. IV antibiotics, close monitoring."
    elif score == 2:
        risk_level = "moderate"
        interpretation = "Moderate CAP (score 2/5): 30-day mortality ~9%"
        recommendation = "Consider hospital admission or supervised outpatient therapy."
    else:
        risk_level = "low"
        interpretation = f"Mild CAP (score {score}/5): 30-day mortality <1–3%"
        recommendation = "Outpatient treatment appropriate in most cases."

    return CalculatorResult(
        calculator="CURB-65",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# CHA₂DS₂-VASc (stroke risk in AF)
# ---------------------------------------------------------------------------

def cha2ds2_vasc(
    chf: bool = False,
    hypertension: bool = False,
    age_gte75: bool = False,
    diabetes: bool = False,
    stroke_or_tia: bool = False,
    vascular_disease: bool = False,
    age_65_74: bool = False,
    female_sex: bool = False,
) -> CalculatorResult:
    """CHA₂DS₂-VASc Score for Stroke Risk in Atrial Fibrillation."""
    breakdown = {
        "Congestive heart failure (1)": int(chf),
        "Hypertension (1)": int(hypertension),
        "Age ≥75 years (2)": 2 if age_gte75 else 0,
        "Diabetes mellitus (1)": int(diabetes),
        "Stroke/TIA/thromboembolism (2)": 2 if stroke_or_tia else 0,
        "Vascular disease (1)": int(vascular_disease),
        "Age 65–74 years (1)": int(age_65_74),
        "Female sex (1)": int(female_sex),
    }
    score = sum(breakdown.values())

    if score >= 2:
        risk_level = "high"
        interpretation = f"High stroke risk (score {score}): annual stroke risk ≥2.2%"
        recommendation = "Oral anticoagulation recommended (DOAC preferred over warfarin unless contraindicated)."
    elif score == 1:
        risk_level = "moderate"
        interpretation = "Moderate stroke risk (score 1): annual stroke risk ~1.3%"
        recommendation = "Anticoagulation should be considered; weigh stroke risk vs bleeding risk."
    else:
        risk_level = "low"
        interpretation = "Low stroke risk (score 0): annual stroke risk ~0%"
        recommendation = "Antithrombotic therapy not recommended for score 0 in males; score 1 in females (sex category alone) also does not require anticoagulation."

    return CalculatorResult(
        calculator="CHA₂DS₂-VASc",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# CHADS₂ (older AF stroke risk — kept for reference)
# ---------------------------------------------------------------------------

def chads2(
    chf: bool = False,
    hypertension: bool = False,
    age_gte75: bool = False,
    diabetes: bool = False,
    stroke_or_tia: bool = False,
) -> CalculatorResult:
    """CHADS₂ Score (older AF stroke risk tool)."""
    breakdown = {
        "Congestive heart failure (1)": int(chf),
        "Hypertension (1)": int(hypertension),
        "Age ≥75 years (1)": int(age_gte75),
        "Diabetes mellitus (1)": int(diabetes),
        "Stroke/TIA history (2)": 2 if stroke_or_tia else 0,
    }
    score = sum(breakdown.values())

    if score >= 2:
        risk_level = "high"
        interpretation = f"High risk (score {score}): annual stroke risk ≥4%"
        recommendation = "Anticoagulation recommended."
    elif score == 1:
        risk_level = "moderate"
        interpretation = "Moderate risk (score 1): annual stroke risk ~2.8%"
        recommendation = "Either aspirin or anticoagulation acceptable; anticoagulation preferred."
    else:
        risk_level = "low"
        interpretation = "Low risk (score 0): annual stroke risk ~1.9%"
        recommendation = "Aspirin 75–325 mg daily or no antithrombotic therapy."

    return CalculatorResult(
        calculator="CHADS₂",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Glasgow Coma Scale
# ---------------------------------------------------------------------------

def glasgow_coma_scale(
    eye_opening: int = 4,    # 1=none,2=to pain,3=to voice,4=spontaneous
    verbal_response: int = 5, # 1=none,2=sounds,3=words,4=confused,5=oriented
    motor_response: int = 6,  # 1=none,2=extension,3=flexion,4=withdrawal,5=localises,6=obeys
) -> CalculatorResult:
    """Glasgow Coma Scale (GCS)."""
    breakdown = {
        "Eye opening (E)": eye_opening,
        "Verbal response (V)": verbal_response,
        "Motor response (M)": motor_response,
    }
    score = sum(breakdown.values())

    if score <= 8:
        risk_level = "high"
        interpretation = f"Severe brain injury (GCS {score}/15)"
        recommendation = "Severe TBI. Immediate airway protection (intubation generally indicated for GCS ≤8). Neurosurgery/ICU."
    elif score <= 12:
        risk_level = "moderate"
        interpretation = f"Moderate brain injury (GCS {score}/15)"
        recommendation = "Moderate TBI. Close neurological monitoring, CT head, neurosurgery consultation."
    else:
        risk_level = "low"
        interpretation = f"Mild brain injury (GCS {score}/15)"
        recommendation = "Mild TBI. CT head based on clinical assessment. Consider observation."

    return CalculatorResult(
        calculator="Glasgow Coma Scale",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# PSI / PORT Score (pneumonia severity index) — simplified version
# ---------------------------------------------------------------------------

def psi_port(
    age_years: int = 50,
    female: bool = False,
    nursing_home: bool = False,
    neoplastic_disease: bool = False,
    liver_disease: bool = False,
    chf: bool = False,
    cerebrovascular: bool = False,
    renal_disease: bool = False,
    altered_mental_status: bool = False,
    respiratory_rate_gte30: bool = False,
    systolic_bp_lt90: bool = False,
    temp_lt35_or_gte40: bool = False,
    pulse_gte125: bool = False,
    ph_lt735: bool = False,
    bun_gte30: bool = False,
    sodium_lt130: bool = False,
    glucose_gte250: bool = False,
    haematocrit_lt30: bool = False,
    po2_lt60: bool = False,
    pleural_effusion: bool = False,
) -> CalculatorResult:
    """Pneumonia Severity Index (PSI/PORT Score) for CAP risk stratification."""
    score = age_years - (10 if female else 0)
    comorbid = {
        "Nursing home resident (+10)": 10 if nursing_home else 0,
        "Neoplastic disease (+30)": 30 if neoplastic_disease else 0,
        "Liver disease (+20)": 20 if liver_disease else 0,
        "Congestive heart failure (+10)": 10 if chf else 0,
        "Cerebrovascular disease (+10)": 10 if cerebrovascular else 0,
        "Renal disease (+10)": 10 if renal_disease else 0,
        "Altered mental status (+20)": 20 if altered_mental_status else 0,
        "RR ≥30 (+20)": 20 if respiratory_rate_gte30 else 0,
        "SBP <90 mmHg (+20)": 20 if systolic_bp_lt90 else 0,
        "Temp <35°C or ≥40°C (+15)": 15 if temp_lt35_or_gte40 else 0,
        "Pulse ≥125 (+10)": 10 if pulse_gte125 else 0,
        "pH <7.35 (+30)": 30 if ph_lt735 else 0,
        "BUN ≥30 mg/dL (+20)": 20 if bun_gte30 else 0,
        "Sodium <130 mEq/L (+20)": 20 if sodium_lt130 else 0,
        "Glucose ≥250 mg/dL (+10)": 10 if glucose_gte250 else 0,
        "Haematocrit <30% (+10)": 10 if haematocrit_lt30 else 0,
        "PO₂ <60 mmHg (+10)": 10 if po2_lt60 else 0,
        "Pleural effusion (+10)": 10 if pleural_effusion else 0,
    }
    score += sum(comorbid.values())
    breakdown = {"Age base score": age_years - (10 if female else 0), **comorbid}

    if score <= 70:
        risk_level = "low"
        psi_class = "I/II"
        interpretation = f"PSI Class {psi_class} (score {score}): 30-day mortality <1%"
        recommendation = "Outpatient treatment appropriate."
    elif score <= 90:
        risk_level = "moderate"
        psi_class = "III"
        interpretation = f"PSI Class {psi_class} (score {score}): 30-day mortality ~2.8%"
        recommendation = "Consider brief inpatient observation or outpatient with close follow-up."
    elif score <= 130:
        risk_level = "high"
        psi_class = "IV"
        interpretation = f"PSI Class {psi_class} (score {score}): 30-day mortality ~8.2%"
        recommendation = "Hospitalise."
    else:
        risk_level = "very_high"
        psi_class = "V"
        interpretation = f"PSI Class {psi_class} (score {score}): 30-day mortality ~29%"
        recommendation = "Hospitalise; consider ICU admission."

    return CalculatorResult(
        calculator="PSI/PORT Score",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# MEWS (Modified Early Warning Score) — simplified
# ---------------------------------------------------------------------------

def mews(
    sbp: int = 120,
    heart_rate: int = 80,
    respiratory_rate: int = 16,
    temperature: float = 37.0,
    avpu: str = "A",   # A=alert, V=voice, P=pain, U=unresponsive
) -> CalculatorResult:
    """Modified Early Warning Score (MEWS) for acute clinical deterioration."""

    def sbp_score(v: int) -> int:
        if v <= 70: return 3
        if v <= 80: return 2
        if v <= 100: return 1
        if v <= 199: return 0
        return 2

    def hr_score(v: int) -> int:
        if v < 40: return 2
        if v < 51: return 1
        if v <= 100: return 0
        if v <= 110: return 1
        if v <= 129: return 2
        return 3

    def rr_score(v: int) -> int:
        if v < 9: return 2
        if v <= 14: return 0
        if v <= 20: return 1
        if v <= 29: return 2
        return 3

    def temp_score(v: float) -> int:
        if v < 35: return 2
        if v <= 38.4: return 0
        return 2

    def avpu_score(v: str) -> int:
        return {"A": 0, "V": 1, "P": 2, "U": 3}.get(v.upper(), 0)

    breakdown = {
        f"Systolic BP {sbp} mmHg": sbp_score(sbp),
        f"Heart rate {heart_rate} bpm": hr_score(heart_rate),
        f"Respiratory rate {respiratory_rate} /min": rr_score(respiratory_rate),
        f"Temperature {temperature}°C": temp_score(temperature),
        f"AVPU {avpu}": avpu_score(avpu),
    }
    score = sum(breakdown.values())

    if score >= 5:
        risk_level = "high"
        interpretation = f"High risk of deterioration (MEWS {score})"
        recommendation = "Immediate senior review. Consider ICU/HDU referral and crash team alert."
    elif score >= 3:
        risk_level = "moderate"
        interpretation = f"Increased risk (MEWS {score})"
        recommendation = "Increase monitoring frequency. Senior nurse/doctor review within 30 minutes."
    else:
        risk_level = "low"
        interpretation = f"Routine monitoring appropriate (MEWS {score})"
        recommendation = "Continue standard observations."

    return CalculatorResult(
        calculator="MEWS",
        score=score,
        interpretation=interpretation,
        risk_level=risk_level,
        recommendation=recommendation,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALCULATOR_REGISTRY: dict[str, callable] = {
    "wells_dvt": wells_dvt,
    "wells_pe": wells_pe,
    "heart_score": heart_score,
    "curb65": curb65,
    "cha2ds2_vasc": cha2ds2_vasc,
    "chads2": chads2,
    "glasgow_coma_scale": glasgow_coma_scale,
    "psi_port": psi_port,
    "mews": mews,
}
