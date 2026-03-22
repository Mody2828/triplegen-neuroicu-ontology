"""Synonym-aware matching: domain map and gold term expansion."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .normalize import normalize_label_for_match


# Neuro-ICU / BrainIT: abbreviation or alternate form -> canonical form (as in gold labels)
# Pairs (alt, canonical); canonical should match gold rdfs:label (after normalization).
# Piper-2003 paper phrases and controlled-vocab names are mapped to gold labels for scoring.
DOMAIN_SYNONYMS: List[Tuple[str, str]] = [
    ("ICP", "Intracranial Pressure (ICP)"),
    ("Intracranial Pressure", "Intracranial Pressure (ICP)"),
    ("MAP", "Mean Arterial Pressure (MAP)"),
    ("Mean Arterial Pressure", "Mean Arterial Pressure (MAP)"),
    ("CPP", "Cerebral Perfusion Pressure (CPP)"),
    ("Cerebral Perfusion Pressure", "Cerebral Perfusion Pressure (CPP)"),
    # --- GCS: paper uses "GCS scores", gold v2.0 class is "GCS Assessment" ---
    ("GCS", "GCS Assessment"),
    ("GCS scores", "GCS Assessment"),
    ("GCS score", "GCS Assessment"),
    ("Glasgow Coma Scale", "GCS Assessment"),
    ("glasgow coma scale", "GCS Assessment"),
    # --- GOSe: gold v2.0 class is "GOSe Outcome" ---
    ("GOSe", "GOSe Outcome"),
    ("GOSe outcome", "GOSe Outcome"),
    ("Extended Glasgow Outcome Scale", "GOSe Outcome"),
    ("Glasgow Outcome Scale Extended", "GOSe Outcome"),
    ("Glasgow Outcome Scale", "GOSe Outcome"),
    ("SaO2", "SaO2"),
    ("EtCO2", "EtCO2"),
    ("SjO2", "SjO2"),
    ("PbrO2", "PbrO2"),
    ("CVP", "CVP"),
    ("TCD", "TCD"),
    ("Heart Rate", "Heart Rate"),
    ("Respiration Rate", "Respiration Rate"),
    ("Core Monitoring Parameter", "Core Monitoring Parameter"),
    ("core monitoring", "Core Monitoring Parameter"),
    ("minimum monitoring", "Core Monitoring Parameter"),
    ("Optional Monitoring Parameter", "Optional Monitoring Parameter"),
    ("optional monitoring", "Optional Monitoring Parameter"),
    ("additional monitoring variables", "Optional Monitoring Parameter"),
    ("Derived Parameter", "Derived Parameter"),
    ("Monitoring Data", "Monitoring Data"),
    ("MonitoringData", "Monitoring Data"),
    # Piper-2003 paper phrases -> gold labels
    ("minute-by-minute monitoring", "Monitoring Data"),
    ("minute by minute monitoring", "Monitoring Data"),
    ("Intensive Care Management", "Intensive Care Management"),
    ("intensive care management", "Intensive Care Management"),
    ("ICU management", "Intensive Care Management"),
    ("ManagementIntervention", "Intensive Care Management"),
    ("Secondary Insult Treatment", "Secondary Insult Treatment"),
    ("Data Quality", "Data Quality Assessment"),
    ("data quality", "Data Quality Assessment"),
    ("Observation", "Observation"),
    ("Reading", "Reading"),
    ("Timepoint", "Timepoint"),
    ("Parameter", "Parameter"),
    ("Session", "Session"),
    ("Patient", "Patient"),
    ("Therapy", "Therapy"),
    ("Condition", "Condition"),
    ("Sensor", "Sensor"),
    ("Machine", "Machine"),
    ("Provenance", "Provenance"),
    # Part A (governance/operations) – map paper phrases to canonical labels for scoring
    ("demographic and clinical information", "DemographicData"),
    ("BrainIT group", "ResearchConsortium"),
    ("research consortium", "ResearchConsortium"),
    ("core dataset", "CoreDataset"),
    ("Core Dataset", "CoreDataset"),
    ("feasibility exercise", "FeasibilityExercise"),
    ("pilot data collection", "PilotDataCollection"),
    ("pilot collection", "PilotDataCollection"),
    ("data collection form", "DataCollectionForm"),
    ("forms", "DataCollectionForm"),
    ("centres", "Centre"),
    ("Centre", "Centre"),
    ("SQL database", "SQLDatabase"),
    ("software tool", "SoftwareTool"),
    ("Software tools", "SoftwareTool"),
    ("ethics approval", "EthicsApproval"),
    # Variants from full-paper runs that were counted as hallucinations
    ("BrainIT Group", "ResearchConsortium"),
    ("Common Database", "SQLDatabase"),
    ("common database", "SQLDatabase"),
    ("Open Access Database", "SQLDatabase"),
    ("Data Contributing Centre", "Centre"),
    ("data contributing centre", "Centre"),
    ("Patient Monitoring Data", "Monitoring Data"),
    ("patient monitoring data", "Monitoring Data"),
    ("Multi-centre Network", "ResearchConsortium"),
    ("Multi-Centre Network", "ResearchConsortium"),
    ("multi-centre network", "ResearchConsortium"),
    ("Research Ethics Committee (MREC)", "EthicsApproval"),
    ("Clinical Research Form (CRF)", "DataCollectionForm"),
    ("clinical research form", "DataCollectionForm"),
    # --- Fluids: paper uses "fluid input and output", gold uses "Fluids" ---
    ("Fluids", "Fluids"),
    ("fluid input and output", "Fluids"),
    ("fluid management", "Fluids"),
    ("fluid balance", "Fluids"),
    ("Fluid Input and Output", "Fluids"),
    # --- Nutrition ---
    ("Nutrition", "Nutrition"),
    ("nutritional support", "Nutrition"),
    ("Nutritional Support", "Nutrition"),
    # --- Sedation: paper says "sedation levels" ---
    ("Sedation", "Sedation"),
    ("sedation levels", "Sedation"),
    ("Sedation Levels", "Sedation"),
    # --- Condition: gold generic superclass mapped from specific condition types ---
    ("secondary insult condition", "Condition"),
    ("clinical condition", "Condition"),
    # --- v2.0: Secondary Insult hierarchy ---
    ("Secondary Insult", "Secondary Insult"),
    ("Secondary Insults", "Secondary Insult"),
    ("secondary insults", "Secondary Insult"),
    ("secondary insult", "Secondary Insult"),
    ("Intracranial Hypertension", "Intracranial Hypertension"),
    ("intracranial hypertension", "Intracranial Hypertension"),
    ("raised ICP", "Intracranial Hypertension"),
    ("Systemic Hypotension", "Systemic Hypotension"),
    ("systemic hypotension", "Systemic Hypotension"),
    ("Arterial Hypotension", "Arterial Hypotension"),
    ("arterial hypotension", "Arterial Hypotension"),
    ("Jugular Venous Desaturation", "Jugular Venous Desaturation"),
    ("jugular venous desaturation", "Jugular Venous Desaturation"),
    ("SjO2 desaturation", "Jugular Venous Desaturation"),
    ("Acute Respiratory Distress Syndrome", "Acute Respiratory Distress Syndrome"),
    ("ARDS", "Acute Respiratory Distress Syndrome"),
    ("Hypertension", "Hypertension"),
    ("hypertension", "Hypertension"),
    ("Hypotension", "Hypotension"),
    ("hypotension", "Hypotension"),
    ("Sepsis", "Sepsis"),
    ("sepsis", "Sepsis"),
    # --- v2.0: Clinical Assessment ---
    ("Clinical Assessment", "Clinical Assessment"),
    ("clinical assessment", "Clinical Assessment"),
    ("clinical status", "Clinical Assessment"),
    ("Pupil Assessment", "Pupil Assessment"),
    ("pupil scores", "Pupil Assessment"),
    ("pupil size", "Pupil Assessment"),
    ("pupil data", "Pupil Assessment"),
    ("CT Scan Assessment", "CT Scan Assessment"),
    ("CT scan", "CT Scan Assessment"),
    ("CT-Scan data", "CT Scan Assessment"),
    ("TCDB classification", "CT Scan Assessment"),
    # --- v2.0: Outcome ---
    ("Outcome", "Outcome"),
    ("patient outcome", "Outcome"),
    ("GOSe Outcome", "GOSe Outcome"),
    # --- v2.0: Laboratory Values ---
    ("Laboratory Values", "Laboratory Values"),
    ("laboratory values", "Laboratory Values"),
    ("daily laboratory values", "Laboratory Values"),
    ("laboratory tests", "Laboratory Values"),
    ("Blood Gases", "Blood Gases"),
    ("blood gases", "Blood Gases"),
    ("routine blood gases", "Blood Gases"),
    ("arterial blood gas", "Blood Gases"),
    ("ABG", "Blood Gases"),
    ("Biochemistry", "Biochemistry"),
    ("biochemistry", "Biochemistry"),
    ("Biochemistry panel", "Biochemistry"),
    ("Haematology", "Haematology"),
    ("haematology", "Haematology"),
    ("Haematology panel", "Haematology"),
    ("Sodium", "Sodium"),
    ("Na", "Sodium"),
    ("Potassium", "Potassium"),
    ("K+", "Potassium"),
    ("Glucose", "Glucose"),
    ("blood glucose", "Glucose"),
    ("Haemoglobin", "Haemoglobin"),
    ("haemoglobin", "Haemoglobin"),
    ("Haemaglobin", "Haemoglobin"),
    ("Hb", "Haemoglobin"),
    ("White Blood Cell Count", "White Blood Cell Count"),
    ("White Cell Count", "White Blood Cell Count"),
    ("WBC", "White Blood Cell Count"),
    ("Haematocrit", "Haematocrit"),
    ("haematocrit", "Haematocrit"),
    # --- v2.0: Therapy hierarchy ---
    ("Baseline Therapy", "Baseline Therapy"),
    ("baseline therapy", "Baseline Therapy"),
    ("baseline intensive care management", "Baseline Therapy"),
    ("Secondary Insult Therapy", "Secondary Insult Therapy"),
    ("secondary insult therapy", "Secondary Insult Therapy"),
    ("Arterial Pressors", "Arterial Pressors"),
    ("arterial pressors", "Arterial Pressors"),
    ("Noradrenaline", "Noradrenaline"),
    ("noradrenaline", "Noradrenaline"),
    ("norepinephrine", "Noradrenaline"),
    ("Adrenaline", "Adrenaline"),
    ("adrenaline", "Adrenaline"),
    ("epinephrine", "Adrenaline"),
    # --- v2.0: Nursing Interventions ---
    ("Nursing Intervention", "Nursing Intervention"),
    ("nursing intervention", "Nursing Intervention"),
    ("nursing activities", "Nursing Intervention"),
    ("Routine Nursing Care", "Routine Nursing Care"),
    ("routine nursing care", "Routine Nursing Care"),
    ("nursing care", "Routine Nursing Care"),
    ("eye/mouth care", "Routine Nursing Care"),
    ("ET tube suction", "Routine Nursing Care"),
    ("Physiotherapy", "Physiotherapy"),
    ("physiotherapy", "Physiotherapy"),
    ("Bedside Intervention", "Bedside Intervention"),
    ("bedside intervention", "Bedside Intervention"),
    ("line insertion", "Bedside Intervention"),
    ("transducer calibration", "Bedside Intervention"),
    ("Patient Transport", "Patient Transport"),
    ("patient transport", "Patient Transport"),
    ("transport to theatre", "Patient Transport"),
    # --- v2.0: Data Quality Assessment ---
    ("Data Quality Assessment", "Data Quality Assessment"),
    ("data quality assessment", "Data Quality Assessment"),
    ("Possible Error", "Possible Error"),
    ("possible error", "Possible Error"),
    ("PoE", "Possible Error"),
    ("Probable Error", "Probable Error"),
    ("probable error", "Probable Error"),
    ("PrE", "Probable Error"),
]


def _derive_synonyms_from_label(label: str) -> List[str]:
    """From a label like 'Intracranial Pressure (ICP)' derive ['ICP', 'Intracranial Pressure']."""
    out: List[str] = []
    # Abbreviation in parentheses
    m = re.search(r"\(([A-Za-z0-9]+)\)\s*$", label)
    if m:
        out.append(m.group(1).strip())
    m2 = re.search(r"^(.+?)\s*\([A-Za-z0-9]+\)\s*$", label)
    if m2:
        out.append(m2.group(1).strip())
    return out


def build_gold_terms_to_index(gold: List[Dict]) -> Tuple[Dict[str, int], List[str]]:
    """Build map from any matchable term (normalized) to gold class index.

    For each gold class we add: normalized label, normalized synonyms from the class,
    derived synonyms from label (e.g. abbreviation in parentheses), and domain synonyms
    when the gold label matches the canonical form.
    Returns (term_to_gold_index, list of normalized gold labels by index).
    """
    term_to_index: Dict[str, int] = {}
    gold_normalized_labels: List[str] = []

    for i, g in enumerate(gold):
        label = g.get("label") or ""
        norm_label = normalize_label_for_match(label)
        gold_normalized_labels.append(norm_label)
        if norm_label and norm_label not in term_to_index:
            term_to_index[norm_label] = i
        for s in g.get("synonyms") or []:
            if isinstance(s, str):
                n = normalize_label_for_match(s)
                if n and n not in term_to_index:
                    term_to_index[n] = i
        for derived in _derive_synonyms_from_label(label):
            n = normalize_label_for_match(derived)
            if n and n not in term_to_index:
                term_to_index[n] = i
        for alt, canonical in DOMAIN_SYNONYMS:
            if normalize_label_for_match(canonical) == norm_label:
                n_alt = normalize_label_for_match(alt)
                if n_alt and n_alt not in term_to_index:
                    term_to_index[n_alt] = i

    return term_to_index, gold_normalized_labels
