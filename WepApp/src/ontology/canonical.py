"""Canonical label normalization and alias resolution for ontology merge/dedupe."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Strip parenthetical acronyms/abbreviations anywhere in the label,
# e.g. "Blood Pressure (BP) Monitoring" → "Blood Pressure Monitoring".
_PAREN_ACRONYM = re.compile(r"\s*\([^)]+\)")

# Spelling variants for canonical key.
_SPELLING_NORMALIZATIONS = (
    (re.compile(r"\bdata\s+sets?\b", re.IGNORECASE), "dataset"),
    (re.compile(r"\bdata\s+set\b", re.IGNORECASE), "dataset"),
    # Merge hyphenated anatomical variants (intra-cranial <-> intracranial, etc.)
    (re.compile(r"\bintra\s*-?\s*cranial\b", re.IGNORECASE), "intracranial"),
    (re.compile(r"\bintra\s*-?\s*ventricular\b", re.IGNORECASE), "intraventricular"),
    (re.compile(r"\bintra\s*-?\s*parenchymal\b", re.IGNORECASE), "intraparenchymal"),
)

# Alias -> canonical label (normalized key for lookup; value is display label).
# Used to merge TBI/CVP/ICP/CPP/GCS variants and parenthetical forms.
CANONICAL_ALIAS_MAP: Dict[str, str] = {
    "tbi": "Traumatic Brain Injury",
    "traumatic brain injury": "Traumatic Brain Injury",
    "traumatic brain injury (tbi)": "Traumatic Brain Injury",
    "cvp": "CVP",
    "central venous pressure": "CVP",
    "central venous pressure (cvp)": "CVP",
    "icp": "Intracranial Pressure (ICP)",
    "intracranial pressure": "Intracranial Pressure (ICP)",
    "intracranial pressure (icp)": "Intracranial Pressure (ICP)",
    "icp intracranial pressure": "Intracranial Pressure (ICP)",
    "cpp": "Cerebral Perfusion Pressure (CPP)",
    "cerebral perfusion pressure": "Cerebral Perfusion Pressure (CPP)",
    "cerebral perfusion pressure (cpp)": "Cerebral Perfusion Pressure (CPP)",
    "hrt": "Heart Rate",
    "heart rate bpm": "Heart Rate",
    "tc": "Temperature",
    "core temperature": "Temperature",
    "peripheral temperature": "Peripheral Temperature",
    "tp": "Peripheral Temperature",
    "brtemp": "Brain Temperature",
    "brain temperature c": "Brain Temperature",
    "ptio2": "PbrO2",
    "brain tissue oxygen": "PbrO2",
    "etco2": "EtCO2",
    "end tidal co2": "EtCO2",
    "bp": "Mean Arterial Pressure (MAP)",
    "bpm": "Mean Arterial Pressure (MAP)",
    "abp": "Mean Arterial Pressure (MAP)",
    "abpm": "Mean Arterial Pressure (MAP)",
    "blood pressure": "Mean Arterial Pressure (MAP)",
    "nibp": "NIBP",
    "non invasive blood pressure": "NIBP",
    "mean arterial blood pressure": "Mean Arterial Pressure (MAP)",
    "systolic blood pressure": "Mean Arterial Pressure (MAP)",
    "prx": "Pressure Reactivity Index (PRx)",
    "pressure reactivity index": "Pressure Reactivity Index (PRx)",
    "pressure reactivity index (prx)": "Pressure Reactivity Index (PRx)",
    "inotropes": "Vasopressors",
    "volume expansion": "Fluids",
    "fluid input output balance": "Fluids",
    "mannitol": "Osmotics",
    "osmotics": "Osmotics",
    "osmotics mannitol": "Osmotics",
    "barbiturates": "Barbiturates",
    "thiopental": "Barbiturates",
    "steroids": "Steroids",
    "corticosteroids": "Steroids",
    "dexamethasone": "Steroids",
    "analgesia": "Analgesia",
    "analgesic": "Analgesia",
    "paralysis": "Paralysis",
    "neuromuscular paralysis": "Paralysis",
    "muscle relaxant": "Paralysis",
    "anti hypertensives": "Anti-hypertensives",
    "antihypertensives": "Anti-hypertensives",
    "anti pyretics": "Anti-pyretics",
    "antipyretics": "Anti-pyretics",
    "paracetamol": "Anti-pyretics",
    "hypothermia": "Hypothermia Therapy",
    "induced hypothermia": "Hypothermia Therapy",
    "therapeutic hypothermia": "Hypothermia Therapy",
    "cooling therapy": "Hypothermia Therapy",
    "hypothermia therapy": "Hypothermia Therapy",
    "hyperventilation therapy": "Secondary Insult Therapy",
    "raised intracranial pressure": "Intracranial Hypertension",
    "increased intracranial pressure": "Intracranial Hypertension",
    "refractory intracranial hypertension": "Intracranial Hypertension",
    # --- Papers 4-10: additional monitoring parameter aliases ---
    "pbto2": "PbrO2",
    "brain tissue oxygen tension": "PbrO2",
    "spo2": "SaO2",
    "pulse oximetry": "SaO2",
    "arterial oxygen saturation": "SaO2",
    "hr": "Heart Rate",
    "abps": "Mean Arterial Pressure (MAP)",
    "bps": "Mean Arterial Pressure (MAP)",
    "arterial blood pressure": "Mean Arterial Pressure (MAP)",
    "diastolic blood pressure": "Mean Arterial Pressure (MAP)",
    # --- Derived indices from papers 5, 8, 9 ---
    "lax": "Derived Parameter",
    "low frequency autoregulation index": "Derived Parameter",
    "rap": "Derived Parameter",
    "rap index": "Derived Parameter",
    "cppopt": "Derived Parameter",
    "optimal cpp": "Derived Parameter",
    "optimal cerebral perfusion pressure": "Derived Parameter",
    "amp": "Derived Parameter",
    "pulse amplitude": "Derived Parameter",
    "ccp": "Derived Parameter",
    "crcp": "Derived Parameter",
    "critical closing pressure": "Derived Parameter",
    "icp dose": "Derived Parameter",
    "pressure time burden": "Derived Parameter",
    "ptd": "Derived Parameter",
    "icp burden": "Derived Parameter",
    "pressure burden": "Derived Parameter",
    "dose of icp": "Derived Parameter",
    # --- Additional conditions from papers 4-10 ---
    "hypoxia": "Secondary Insult",
    "cerebral ischemia": "Condition",
    "cerebral hypoperfusion": "Condition",
    "delayed cerebral ischemia": "Condition",
    "vasospasm": "Condition",
    "cerebral edema": "Condition",
    "brain swelling": "Condition",
    "diffuse axonal injury": "Condition",
    "dai": "Condition",
    "mass lesion": "Condition",
    "impaired cerebral autoregulation": "Condition",
    "hypotensive event": "Arterial Hypotension",
    "eusig defined hypotensive event": "Arterial Hypotension",
    # --- Additional therapy from papers 6, 8 ---
    "decompressive craniectomy": "Decompressive Craniectomy",
    "hemicraniectomy": "Decompressive Craniectomy",
    "hyperosmolar therapy": "Osmotics",
    "hypertonic saline": "Osmotics",
    "csf drainage": "Extra Ventricular Drain Placement",
    "cerebrospinal fluid drainage": "Extra Ventricular Drain Placement",
    "evd": "Extra Ventricular Drain Placement",
    "external ventricular drain": "Extra Ventricular Drain Placement",
    "ventriculostomy": "Extra Ventricular Drain Placement",
    "lumbar drain": "Extra Ventricular Drain Placement",
    "icp sensor placement": "ICP Sensor Placement",
    "icp monitor insertion": "ICP Sensor Placement",
    "icp bolt": "ICP Sensor Placement",
    "icp catheter": "ICP Sensor Placement",
    "evacuation of mass lesion": "Evacuation of Mass Lesion",
    "craniotomy": "Evacuation of Mass Lesion",
    "skull fracture elevation": "Skull Fracture Elevation",
    "removal of foreign body": "Removal of Foreign Body",
    "anterior fossa repair": "Anterior Fossa Repair",
    "csf leak repair": "Anterior Fossa Repair",
    "surgical procedure": "Surgical Procedure",
    "thiopental coma": "Barbiturates",
    "barbiturate coma": "Barbiturates",
    "moderate hypocapnia": "Secondary Insult Therapy",
    "normocapnia": "Baseline Therapy",
    "normothermia": "Baseline Therapy",
    "prophylactic hyperventilation": "Secondary Insult Therapy",
    "cpp targeted therapy": "Therapy",
    "icp targeted therapy": "Therapy",
    "neuroprotective drug": "Therapy",
    "fluid resuscitation": "Fluids",
    "pressors": "Vasopressors",
    # --- Additional outcome / assessment ---
    "favorable outcome": "GOSe Outcome",
    "unfavorable outcome": "GOSe Outcome",
    "six month outcome": "GOSe Outcome",
    "marshall classification": "CT Scan Assessment",
    "marshall score": "CT Scan Assessment",
    "anisocoria": "Pupil Assessment",
    "gcs motor": "GCS Assessment",
    "gcs eye": "GCS Assessment",
    "gcs verbal": "GCS Assessment",
    "gcsv": "GCS Assessment",
    "gms": "GCS Assessment",
    "glasgow motor score": "GCS Assessment",
    "brain death": "Outcome",
    # --- Additional lab values ---
    "pao2": "Blood Gases",
    "pco2": "Blood Gases",
    "paco2": "Blood Gases",
    "s100b": "Biochemistry",
    # --- Data quality ---
    "artefact": "Data Quality Assessment",
    "artifact": "Data Quality Assessment",
    "physiological artefact": "Data Quality Assessment",
    # --- Nursing ---
    "endotracheal suction": "Routine Nursing Care",
    "patient turning": "Routine Nursing Care",
    "blood sampling": "Bedside Intervention",
    # --- GCS: paper uses "GCS scores", gold v2.0 class is "GCS Assessment" ---
    "gcs": "GCS Assessment",
    "gcs score": "GCS Assessment",
    "gcs scores": "GCS Assessment",
    "glasgow coma scale": "GCS Assessment",
    "glasgow coma scale score": "GCS Assessment",
    # --- GOSe: paper uses "GOSe outcome", gold v2.0 class is "GOSe Outcome" ---
    "gose": "GOSe Outcome",
    "gose outcome": "GOSe Outcome",
    "glasgow outcome scale extended": "GOSe Outcome",
    "extended glasgow outcome scale": "GOSe Outcome",
    "glasgow outcome scale": "GOSe Outcome",
    "map": "Mean Arterial Pressure (MAP)",
    "mean arterial pressure": "Mean Arterial Pressure (MAP)",
    "mean arterial pressure (map)": "Mean Arterial Pressure (MAP)",
    # --- Fluids: paper uses "fluid input and output", gold uses "Fluids" ---
    "fluids": "Fluids",
    "fluid": "Fluids",
    "fluid input and output": "Fluids",
    "fluid input": "Fluids",
    "fluid output": "Fluids",
    "fluid management": "Fluids",
    "fluid balance": "Fluids",
    "fluids input and output": "Fluids",
    # --- Nutrition: short form vs paper phrasing ---
    "nutrition": "Nutrition",
    "nutritional support": "Nutrition",
    "nutritional intake": "Nutrition",
    # --- Sedation: paper says "sedation levels", gold says "Sedation" ---
    "sedation": "Sedation",
    "sedation levels": "Sedation",
    "sedation level": "Sedation",
    # --- Condition: gold generic superclass for secondary insults / clinical conditions ---
    "condition": "Condition",
    "clinical condition": "Condition",
    "medical condition": "Condition",
    # --- Secondary Insult hierarchy (v2.0) ---
    "secondary insult": "Secondary Insult",
    "secondary insults": "Secondary Insult",
    "intracranial hypertension": "Intracranial Hypertension",
    "raised icp": "Intracranial Hypertension",
    "systemic hypotension": "Systemic Hypotension",
    "arterial hypotension": "Arterial Hypotension",
    "jugular venous desaturation": "Jugular Venous Desaturation",
    "sjo2 desaturation": "Jugular Venous Desaturation",
    "ards": "Acute Respiratory Distress Syndrome",
    "acute respiratory distress syndrome": "Acute Respiratory Distress Syndrome",
    "hypertension": "Hypertension",
    "hypotension": "Hypotension",
    "sepsis": "Sepsis",
    # --- Clinical Assessment (v2.0) ---
    "clinical assessment": "Clinical Assessment",
    "pupil assessment": "Pupil Assessment",
    "pupil scores": "Pupil Assessment",
    "pupil data": "Pupil Assessment",
    "pupil size": "Pupil Assessment",
    "ct scan assessment": "CT Scan Assessment",
    "ct scan data": "CT Scan Assessment",
    "ct-scan data": "CT Scan Assessment",
    "ct scans": "CT Scan Assessment",
    "ct classification": "CT Scan Assessment",
    "tcdb classification": "CT Scan Assessment",
    # --- Outcome (v2.0) ---
    "outcome": "Outcome",
    # --- Laboratory Values (v2.0) ---
    "laboratory values": "Laboratory Values",
    "lab values": "Laboratory Values",
    "laboratory tests": "Laboratory Values",
    "blood gases": "Blood Gases",
    "arterial blood gas": "Blood Gases",
    "abg": "Blood Gases",
    "routine blood gases": "Blood Gases",
    "biochemistry": "Biochemistry",
    "biochemistry panel": "Biochemistry",
    "haematology": "Haematology",
    "haematology panel": "Haematology",
    "hematology": "Haematology",
    "sodium": "Sodium",
    "na": "Sodium",
    "serum sodium": "Sodium",
    "potassium": "Potassium",
    "k+": "Potassium",
    "serum potassium": "Potassium",
    "glucose": "Glucose",
    "blood glucose": "Glucose",
    "haemoglobin": "Haemoglobin",
    "hemoglobin": "Haemoglobin",
    "haemaglobin": "Haemoglobin",
    "hb": "Haemoglobin",
    "white blood cell count": "White Blood Cell Count",
    "white cell count": "White Blood Cell Count",
    "wbc": "White Blood Cell Count",
    "haematocrit": "Haematocrit",
    "hematocrit": "Haematocrit",
    # --- Therapy hierarchy (v2.0) ---
    "baseline therapy": "Baseline Therapy",
    "baseline medical management": "Baseline Therapy",
    "baseline intensive care": "Baseline Therapy",
    "secondary insult therapy": "Secondary Insult Therapy",
    "secondary insult treatment": "Secondary Insult Treatment",
    "arterial pressors": "Arterial Pressors",
    "noradrenaline": "Noradrenaline",
    "norepinephrine": "Noradrenaline",
    "adrenaline": "Adrenaline",
    "epinephrine": "Adrenaline",
    # --- Nursing Interventions (v2.0) ---
    "nursing intervention": "Nursing Intervention",
    "nursing interventions": "Nursing Intervention",
    "routine nursing care": "Routine Nursing Care",
    "nursing care": "Routine Nursing Care",
    "physiotherapy": "Physiotherapy",
    "bedside intervention": "Bedside Intervention",
    "line insertion": "Bedside Intervention",
    "transducer calibration": "Bedside Intervention",
    "patient transport": "Patient Transport",
    # --- Monitoring hierarchy (v2.0) ---
    "core monitoring parameter": "Core Monitoring Parameter",
    "core monitoring": "Core Monitoring Parameter",
    "optional monitoring parameter": "Optional Monitoring Parameter",
    "optional monitoring": "Optional Monitoring Parameter",
    "derived parameter": "Derived Parameter",
    # --- Observation framework (Moss 2013 / v2.0) ---
    "session": "Session",
    "timepoint": "Timepoint",
    "observation": "Observation",
    "parameter": "Parameter",
    "sensor": "Sensor",
    # --- Data Quality (v2.0) ---
    "data quality assessment": "Data Quality Assessment",
    "data quality": "Data Quality Assessment",
    "possible error": "Possible Error",
    "poe": "Possible Error",
    "probable error": "Probable Error",
    "pre": "Probable Error",
    # --- v3.0 newly promoted gold classes ---
    "demographic data": "Demographic Data",
    "demographic information": "Demographic Data",
    "demographics": "Demographic Data",
    "cardiac output": "Cardiac Output",
    "co": "Cardiac Output",
    "peripheral temperature": "Peripheral Temperature",
    "pressure reactivity index": "Pressure Reactivity Index (PRx)",
    "pressure reactivity index (prx)": "Pressure Reactivity Index (PRx)",
    "prx index": "Pressure Reactivity Index (PRx)",
    "intensive care management": "Intensive Care Management",
    "icu management": "Intensive Care Management",
    "monitoring data": "Monitoring Data",
    "monitoring datas": "Monitoring Data",
    # Clinical synonym merges (Phase 25.5)
    "btf guidelines": "TBI Guidelines",
    "brain trauma foundation guidelines": "TBI Guidelines",
    "tbi guidelines": "TBI Guidelines",
    "adult respiratory distress syndrome": "Acute Respiratory Distress Syndrome",
    "adult respiratory distress syndrome (ards)": "Acute Respiratory Distress Syndrome",
    "gos outcome": "GOSe Outcome",
    "gos": "GOSe Outcome",
    "glasgow outcome scale score": "GOSe Outcome",
    # LLM often uses these verbose synonyms for Monitoring Data
    "intensive care monitoring": "Monitoring Data",
    "bedside monitoring data": "Monitoring Data",
    "high-resolution monitoring data": "Monitoring Data",
    "minute by minute monitoring data": "Monitoring Data",
    "real-time monitoring data": "Monitoring Data",
    "therapy": "Therapy",
    "patient": "Patient",
}


def _key_for_lookup(s: str) -> str:
    """Normalize for alias map lookup: lowercase, collapse punct to space, collapse whitespace."""
    if not s:
        return ""
    s = (s or "").strip().lower()
    s = _PAREN_ACRONYM.sub(" ", s).strip()
    for pat, repl in _SPELLING_NORMALIZATIONS:
        s = pat.sub(repl, s)
    return " ".join(re.sub(r"[^a-z0-9\s]+", " ", s).split())


def _singularize_token(tok: str) -> str:
    """Simple singular form: strip trailing 's' for common plurals; leave acronyms (GCS, ICP) intact."""
    if not tok or len(tok) <= 2:
        return tok
    if tok.isupper() and len(tok) <= 4:  # acronym
        return tok
    if tok.endswith("sses"):
        return tok[:-4] + "ss"  # glasses -> glass
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"  # therapies -> therapy
    if tok.endswith("tions"):
        return tok[:-1]  # interventions -> intervention (not caught by 'ns')
    if tok.endswith("ults"):
        return tok[:-1]  # insults -> insult
    if tok.endswith("s") and tok[-2] not in "suxz":
        return tok[:-1]  # patients -> patient, devices -> device
    return tok


# Hyphen-joined prefixes that should be collapsed for key normalization.
_HYPHEN_COLLAPSE = re.compile(
    r"\b(hold)[\s-](down)\b"
    r"|\b(neuro)[\s-](intensive)\b"
    r"|\b(anti)[\s-](hypertensive|pyretic|epileptic|coagulant)s?\b"
    r"|\b(non)[\s-](invasive)\b"
    r"|\b(intra)[\s-](cranial|ventricular|parenchymal)\b",
    re.IGNORECASE,
)


def _collapse_hyphens(s: str) -> str:
    """Collapse common hyphen/space-separated compound words."""
    return _HYPHEN_COLLAPSE.sub(lambda m: "".join(g for g in m.groups() if g), s)


def _tokens_for_key(s: str) -> list[str]:
    """Tokenize for canonical key (alphanumeric tokens)."""
    s = _collapse_hyphens((s or "").lower())
    return [x for x in re.findall(r"[A-Za-z0-9]+", s) if x]


def canonical_key(label: str) -> str:
    """
    Normalize label for deduplication: strip parenthetical acronyms, spelling variants,
    singular/plural (patients->patient), lowercase, collapse non-alphanumeric, collapse hyphens.
    """
    if not label:
        return ""
    s = (label or "").strip()
    s = _PAREN_ACRONYM.sub(" ", s).strip()
    s_lower = s.lower()
    for pat, replacement in _SPELLING_NORMALIZATIONS:
        s_lower = pat.sub(replacement, s_lower)
    toks = _tokens_for_key(s_lower)
    # Singularize last token for plural merge (head injured patients ~ head injured patient)
    # Also singularize penultimate if it looks like a plural noun (e.g. "bedside interventions data")
    if toks:
        toks[-1] = _singularize_token(toks[-1])
        if len(toks) >= 2:
            toks[-2] = _singularize_token(toks[-2])
    return " ".join(toks)


def resolve_to_canonical_label(label: str) -> Tuple[str, bool]:
    """
    Map label to canonical form using alias map. Returns (canonical_label, was_mapped).
    If mapped, caller should store original as alias.
    """
    if not label or not label.strip():
        return (label or "").strip(), False
    raw = (label or "").strip()
    key = _key_for_lookup(raw)
    if not key:
        return raw, False
    canonical = CANONICAL_ALIAS_MAP.get(key)
    if canonical:
        return canonical, True
    # Also try without parenthetical (e.g. "Central Venous Pressure (CVP)" -> key "central venous pressure cvp")
    # The _key_for_lookup already strips parens, so "central venous pressure (cvp)" -> "central venous pressure"
    return raw, False
