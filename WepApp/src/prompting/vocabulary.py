"""Controlled vocabulary: Part A (governance), Part B (clinical core), provenance (Moss 2013)."""

from __future__ import annotations

# Relation minimalism: when in restricted mode, only these relation labels are allowed (from gold or below).
# Domain and range are required for every relation (enforced in schema.filter_parsed_to_vocabulary).
# Aligned with brainit_core_2003.ttl gold ontology v2.0.
STRICT_RELATIONS_CLINICAL_REFERENCE: tuple[str, ...] = (
    "includes",
    "hasMonitoringData",
    "receivesTherapy",
    "targetsCondition",
    "monitoringIndicatesCondition",
    "hasDemographicData",
    "indicatesCondition",
    "secondary_to",
    "is_a",
    # v2.0 gold schema relations
    "hasClinicalAssessment",
    "hasOutcome",
    "hasLaboratoryValue",
    "hasNursingIntervention",
    "hasSession",
    "hasTimepoint",
    "hasObservation",
    "measuresParameter",
    "producedBySensor",
    "hasQualityAssessment",
    "associatedWithCondition",
    "affectedByTreatment",
    # v3.0
    "hasSurgicalProcedure",
    # v3.1 — papers 3-11
    "hasInjuryMechanism",
    "triggersIntervention",
    "hasEUSIGGrade",
    "hasMortality",
    "hasGuidelineAdherence",
)

# --- Part B: Clinical core classes (v2.0 combined: Piper 2003 + Moss 2013) ---
ALLOWED_CLASSES_CORE: frozenset[str] = frozenset({
    "Patient",
    "Demographic Data",
    "Session",
    "Timepoint",
    "Monitoring Data",
    "Core Monitoring Parameter",
    "Optional Monitoring Parameter",
    "Derived Parameter",
    "Heart Rate",
    "Respiration Rate",
    "Mean Arterial Pressure (MAP)",
    "Intracranial Pressure (ICP)",
    "SaO2",
    "Temperature",
    "CVP",
    "EtCO2",
    "SjO2",
    "TCD",
    "PbrO2",
    "Brain Temperature",
    "Microdialysis",
    "Cerebral Perfusion Pressure (CPP)",
    "Peripheral Temperature",
    "Cardiac Output",
    "NIBP",
    "Pressure Reactivity Index (PRx)",
    "Therapy",
    "Baseline Therapy",
    "Secondary Insult Therapy",
    "Ventilation",
    "Sedation",
    "Fluids",
    "Nutrition",
    "Vasopressors",
    "Antibiotics",
    "Noradrenaline",
    "Adrenaline",
    "Secondary Insult Treatment",
    "Arterial Pressors",
    "Analgesia",
    "Paralysis",
    "Anti-hypertensives",
    "Anti-pyretics",
    "Hypothermia Therapy",
    "Osmotics",
    "Barbiturates",
    "Steroids",
    "Surgical Procedure",
    "ICP Sensor Placement",
    "Evacuation of Mass Lesion",
    "Skull Fracture Elevation",
    "Extra Ventricular Drain Placement",
    "Decompressive Craniectomy",
    "Removal of Foreign Body",
    "Anterior Fossa Repair",
    "Condition",
    "Secondary Insult",
    "Intracranial Hypertension",
    "Systemic Hypotension",
    "Arterial Hypotension",
    "Jugular Venous Desaturation",
    "Acute Respiratory Distress Syndrome",
    "Hypertension",
    "Hypotension",
    "Sepsis",
    "Clinical Assessment",
    "GCS Assessment",
    "Pupil Assessment",
    "CT Scan Assessment",
    "Outcome",
    "GOSe Outcome",
    "Laboratory Values",
    "Blood Gases",
    "Biochemistry",
    "Haematology",
    "Sodium",
    "Potassium",
    "Glucose",
    "Haemoglobin",
    "White Blood Cell Count",
    "Haematocrit",
    "Nursing Intervention",
    "Routine Nursing Care",
    "Physiotherapy",
    "Bedside Intervention",
    "Patient Transport",
    "Intensive Care Management",
    "Observation",
    "Parameter",
    "Sensor",
    "Data Quality Assessment",
    "Possible Error",
    "Probable Error",
    # v3.0 — papers 3-11 additions
    "Systolic Blood Pressure (SBP)",
    "Diastolic Blood Pressure (DBP)",
    "Pulse Pressure",
    "Electrocardiogram (ECG)",
    "Pressure-Time Dose",
    "Hypotensive Event",
    "Cerebrovascular Autoregulation",
    "EUSIG Grade",
    "Therapy Intensity Level (TIL)",
    "Guideline Adherence",
    "Injury Mechanism",
    "Hemicraniectomy",
    "Bifrontal Craniectomy",
    "Cranioplasty",
    "Head Elevation",
    "Mortality",
    "Artefact",
})

ALLOWED_RELATIONS_CORE: frozenset[str] = frozenset({
    "hasDemographicData",
    "hasMonitoringData",
    "receivesTherapy",
    "targetsCondition",
    "indicatesCondition",
    "includes",
    "secondary_to",
    # v2.0
    "hasClinicalAssessment",
    "hasOutcome",
    "hasLaboratoryValue",
    "hasNursingIntervention",
    "hasSession",
    "hasTimepoint",
    "hasObservation",
    "measuresParameter",
    "producedBySensor",
    "hasQualityAssessment",
    "associatedWithCondition",
    "affectedByTreatment",
    # v3.0
    "hasSurgicalProcedure",
    "monitoringIndicatesCondition",
    # v3.1 — papers 3-11
    "hasInjuryMechanism",
    "triggersIntervention",
    "hasEUSIGGrade",
    "hasMortality",
    "hasGuidelineAdherence",
})

# Extraction-friendly relation labels (paper wording) for prompts when gold is BrainIT clinical.
# Injected into prompts instead of raw gold schema labels; filter maps these to gold via RELATION_ALIASES_CORE.
EXTRACTION_RELATION_LABELS: tuple[str, ...] = (
    "includes",
    "has_source",
    "treats",
    "secondary_to",
    "has_target",
    "is_a",
    # v2.0 new relations (using readable space-separated form for LLM prompts)
    "has clinical assessment",
    "has outcome",
    "has laboratory value",
    "has nursing intervention",
    "has session",
    "has timepoint",
    "has observation",
    "measures parameter",
    "produced by sensor",
    "has quality assessment",
    "associated with condition",
    "affected by treatment",
    "monitoring indicates condition",
    "targets condition",
    "receives therapy",
    "has monitoring data",
    "has surgical procedure",
    # v3.1 — papers 3-11
    "has injury mechanism",
    "triggers intervention",
    "has EUSIG grade",
    "has mortality",
    "has guideline adherence",
)

# LLM often returns these variants; map to canonical for filtering
RELATION_ALIASES_CORE: dict[str, str] = {
    "receiveTherapy": "receivesTherapy",
    "receivesTherapy": "receivesTherapy",
    "receives therapy": "receivesTherapy",
    "targetCondition": "targetsCondition",
    "targetsCondition": "targetsCondition",
    "targets condition": "targetsCondition",
    "treats": "targetsCondition",
    "indicatesCondition": "indicatesCondition",
    "monitoringIndicatesCondition": "monitoringIndicatesCondition",
    "monitoring indicates condition": "monitoringIndicatesCondition",
    "hasMonitoringData": "hasMonitoringData",
    "has monitoring data": "hasMonitoringData",
    "has_source": "hasMonitoringData",
    "hasDemographicData": "hasDemographicData",
    "has_target": "targetsCondition",
    "includes": "includes",
    "secondary_to": "secondary_to",
    # v2.0 aliases (space-separated → camelCase canonical)
    "hasClinicalAssessment": "hasClinicalAssessment",
    "has clinical assessment": "hasClinicalAssessment",
    "hasOutcome": "hasOutcome",
    "has outcome": "hasOutcome",
    "hasLaboratoryValue": "hasLaboratoryValue",
    "has laboratory value": "hasLaboratoryValue",
    "hasNursingIntervention": "hasNursingIntervention",
    "has nursing intervention": "hasNursingIntervention",
    "hasSession": "hasSession",
    "has session": "hasSession",
    "hasTimepoint": "hasTimepoint",
    "has timepoint": "hasTimepoint",
    "hasObservation": "hasObservation",
    "has observation": "hasObservation",
    "measuresParameter": "measuresParameter",
    "measures parameter": "measuresParameter",
    "producedBySensor": "producedBySensor",
    "produced by sensor": "producedBySensor",
    "hasQualityAssessment": "hasQualityAssessment",
    "has quality assessment": "hasQualityAssessment",
    "associatedWithCondition": "associatedWithCondition",
    "associated with condition": "associatedWithCondition",
    "affectedByTreatment": "affectedByTreatment",
    "affected by treatment": "affectedByTreatment",
    "hasSurgicalProcedure": "hasSurgicalProcedure",
    "has surgical procedure": "hasSurgicalProcedure",
    # v3.1 — papers 3-11
    "hasInjuryMechanism": "hasInjuryMechanism",
    "has injury mechanism": "hasInjuryMechanism",
    "injury mechanism": "hasInjuryMechanism",
    "triggersIntervention": "triggersIntervention",
    "triggers intervention": "triggersIntervention",
    "triggers": "triggersIntervention",
    "hasEUSIGGrade": "hasEUSIGGrade",
    "has EUSIG grade": "hasEUSIGGrade",
    "EUSIG grade": "hasEUSIGGrade",
    "hasMortality": "hasMortality",
    "has mortality": "hasMortality",
    "hasGuidelineAdherence": "hasGuidelineAdherence",
    "has guideline adherence": "hasGuidelineAdherence",
    "guideline adherence": "hasGuidelineAdherence",
    # Common LLM-generated variants → canonical forms
    "monitors": "hasMonitoringData",
    "monitored by": "hasMonitoringData",
    "monitored with": "hasMonitoringData",
    "measured by": "measuresParameter",
    "measured with": "measuresParameter",
    "indicates": "monitoringIndicatesCondition",
    "indicates condition": "monitoringIndicatesCondition",
    "causes": "associatedWithCondition",
    "caused by": "associatedWithCondition",
    "associated with": "associatedWithCondition",
    "related to": "associatedWithCondition",
    "manages": "targetsCondition",
    "managed by": "receivesTherapy",
    "administered to": "receivesTherapy",
    "receives": "receivesTherapy",
    "undergoes": "receivesTherapy",
    "produces": "producedBySensor",
    "recorded by": "producedBySensor",
    "records": "hasMonitoringData",
    "detects": "monitoringIndicatesCondition",
    "predicts": "monitoringIndicatesCondition",
    "affects": "affectedByTreatment",
    "influences": "affectedByTreatment",
    "assessed by": "hasClinicalAssessment",
    "assessed with": "hasClinicalAssessment",
    "evaluates": "hasClinicalAssessment",
    "results in": "hasOutcome",
    "leads to": "hasOutcome",
    "has treatment": "receivesTherapy",
    "has therapy": "receivesTherapy",
    "has intervention": "hasNursingIntervention",
    "has assessment": "hasClinicalAssessment",
    "has condition": "associatedWithCondition",
    "has procedure": "hasSurgicalProcedure",
    "has sensor": "producedBySensor",
    "has data": "hasMonitoringData",
    "has demographic": "hasDemographicData",
    "quantifies": "measuresParameter",
    "provides": "hasMonitoringData",
    "derived from": "includes",
    "part of": "includes",
    "consists of": "includes",
    "has component": "includes",
    "type of": "is_a",
    "classified as": "is_a",
    "categorized as": "is_a",
    "subclass of": "is_a",
}

# --- Part A: Governance/operations (Piper 2003 concept, network, forms, DB, ethics) ---
ALLOWED_CLASSES_GOVERNANCE: frozenset[str] = frozenset({
    "ResearchConsortium",
    "CoreDataset",
    "FeasibilityExercise",
    "PilotDataCollection",
    "DataCollectionForm",
    "Centre",
    "DemographicData",
    "MonitoringData",
    "SQLDatabase",
    "SoftwareTool",
    "EthicsApproval",
})

ALLOWED_RELATIONS_GOVERNANCE: frozenset[str] = frozenset({
    "definesCoreDataset",
    "conductsFeasibilityExercise",
    "runsPilotCollection",
    "distributesForm",
    "participates",
    "includesDemographicData",
    "includesMonitoringData",
    "storesIn",
    "supportedBy",
    "hasEthicsApproval",
})

# --- Provenance / Trusting ICU (Moss 2013) ---
ALLOWED_CLASSES_PROVENANCE: frozenset[str] = frozenset({
    "Patient",
    "Session",
    "Timepoint",
    "Reading",
    "Parameter",
    "Observation",
    "Sensor",
    "Machine",
    "Provenance",
    "MeasurementCapability",
    "SamplingRate",
    "DataQuality",
    "RDF Data",
})

# Keywords that indicate provenance/RDF input → use provenance pool & vocab
PROVENANCE_KEYWORDS = frozenset({
    "rdf", "ssn", "provenance", "sensor", "sampling rate",
    "observation value", "trust", "data quality", "annotation",
})

# Keywords that indicate Part A (governance/operations) → use governance pool & prompt
GOVERNANCE_KEYWORDS = frozenset({
    "research consortium", "consortium", "brainit group", "open collaborative",
    "core dataset", "core dataset definition", "feasibility exercise", "feasibility",
    "pilot collection", "pilot data collection", "data collection form", "forms",
    "centres", "centre", "multi-centre", "multicentre", "22 brainit",
    "sql database", "sql database to hold", "software tool", "software tools",
    "ethics approval", "ethics", "meetings", "distributed", "paper based pilot",
})

# Keywords in example text that disqualify it for core-dataset (Part B) demos
CORE_DISALLOWED_IN_EXAMPLE = frozenset({
    "rdf", "ssn", "provenance", "sensor", "sampling rate", "observation value",
})


CLASS_SYNONYM_MAP: dict[str, str] = {
    # --- Monitoring parameters (all papers) ---
    "HRT": "Heart Rate",
    "HR": "Heart Rate",
    "heart rate (bpm)": "Heart Rate",
    "Tc": "Temperature",
    "Core Temperature": "Temperature",
    "core temperature": "Temperature",
    "Tp": "Temperature",
    "peripheral temperature": "Peripheral Temperature",
    "brTemp": "Brain Temperature",
    "brain temperature (°c)": "Brain Temperature",
    "PtiO2": "PbrO2",
    "ptio2": "PbrO2",
    "PbtO2": "PbrO2",
    "pbto2": "PbrO2",
    "brain tissue oxygen": "PbrO2",
    "brain tissue oxygen tension": "PbrO2",
    "ETCO2": "EtCO2",
    "end tidal co2": "EtCO2",
    "SpO2": "SaO2",
    "spo2": "SaO2",
    "pulse oximetry": "SaO2",
    "arterial oxygen saturation": "SaO2",
    "CO": "Cardiac Output",
    "cardiac output": "Cardiac Output",
    "FiO2": "Ventilation",
    "fio2": "Ventilation",
    "fraction of inspired oxygen": "Ventilation",
    "NIRS": "Optional Monitoring Parameter",
    "near-infrared spectroscopy": "Optional Monitoring Parameter",

    # --- Blood pressure variants (all papers) ---
    "BP": "Mean Arterial Pressure (MAP)",
    "BPm": "Mean Arterial Pressure (MAP)",
    "ABP": "Mean Arterial Pressure (MAP)",
    "ABPm": "Mean Arterial Pressure (MAP)",
    "ABPs": "Systolic Blood Pressure (SBP)",
    "BPs": "Systolic Blood Pressure (SBP)",
    "blood pressure": "Mean Arterial Pressure (MAP)",
    "NIBP": "NIBP",
    "non-invasive blood pressure": "NIBP",
    "mean arterial blood pressure": "Mean Arterial Pressure (MAP)",
    "systolic blood pressure": "Systolic Blood Pressure (SBP)",
    "diastolic blood pressure": "Diastolic Blood Pressure (DBP)",
    "arterial blood pressure": "Mean Arterial Pressure (MAP)",

    # --- Derived parameters (papers 2, 5, 8, 9) ---
    "PrX": "Pressure Reactivity Index (PRx)",
    "PRx": "Pressure Reactivity Index (PRx)",
    "pressure reactivity index": "Pressure Reactivity Index (PRx)",
    "LAx": "Derived Parameter",
    "low-frequency autoregulation index": "Derived Parameter",
    "RAP": "Derived Parameter",
    "RAP index": "Derived Parameter",
    "CPPopt": "Derived Parameter",
    "optimal CPP": "Derived Parameter",
    "optimal cerebral perfusion pressure": "Derived Parameter",
    "AMP": "Derived Parameter",
    "pulse amplitude": "Derived Parameter",
    "CCP": "Derived Parameter",
    "CrCP": "Derived Parameter",
    "critical closing pressure": "Derived Parameter",
    "ICP dose": "Pressure-Time Dose",
    "pressure-time burden": "Pressure-Time Dose",
    "pressure time burden": "Pressure-Time Dose",
    "PTD": "Pressure-Time Dose",
    "ICP burden": "Pressure-Time Dose",
    "pressure burden": "Pressure-Time Dose",
    "dose of ICP": "Pressure-Time Dose",
    "pressure-time dose": "Pressure-Time Dose",
    "ICP dose burden": "Pressure-Time Dose",

    # --- GCS variants (all papers) ---
    "Glasgow Coma Score": "GCS Assessment",
    "Glasgow Coma Scale": "GCS Assessment",
    "GCS": "GCS Assessment",
    "gcs scores": "GCS Assessment",
    "GCS Motor": "GCS Assessment",
    "GCS Eye": "GCS Assessment",
    "GCS Verbal": "GCS Assessment",
    "GCSv": "GCS Assessment",
    "GMS": "GCS Assessment",
    "Glasgow Motor Score": "GCS Assessment",

    # --- Outcome variants (all papers) ---
    "GOSe": "GOSe Outcome",
    "GOS": "GOSe Outcome",
    "GOSE": "GOSe Outcome",
    "Glasgow Outcome Score": "GOSe Outcome",
    "Glasgow Outcome Scale": "GOSe Outcome",
    "Extended Glasgow Outcome Scale": "GOSe Outcome",
    "favorable outcome": "GOSe Outcome",
    "unfavorable outcome": "GOSe Outcome",
    "six-month outcome": "GOSe Outcome",

    # --- CT / imaging (papers 1, 3, 6) ---
    "CT scan": "CT Scan Assessment",
    "CT": "CT Scan Assessment",
    "TCDB classification": "CT Scan Assessment",
    "Marshall classification": "CT Scan Assessment",
    "Marshall score": "CT Scan Assessment",
    "TCDB nomenclature": "CT Scan Assessment",

    # --- Pupil assessment (all papers) ---
    "pupil size": "Pupil Assessment",
    "pupil reactivity": "Pupil Assessment",
    "pupil size and reactivity": "Pupil Assessment",
    "anisocoria": "Pupil Assessment",
    "bilateral fixed dilated pupils": "Pupil Assessment",

    # --- Therapy: Vasopressors / inotropes (papers 1, 2, 6) ---
    "Inotropes": "Vasopressors",
    "inotropes": "Vasopressors",
    "arterial pressors": "Arterial Pressors",
    "pressors": "Vasopressors",

    # --- Therapy: Fluids (papers 1, 2, 6) ---
    "Volume expansion": "Fluids",
    "volume expansion": "Fluids",
    "Fluid Input/Output Balance": "Fluids",
    "fluid input/output balance": "Fluids",
    "fluid input and output": "Fluids",
    "fluid resuscitation": "Fluids",

    # --- Therapy: Baseline (papers 1, 2, 6) ---
    "neuromuscular paralysis": "Paralysis",
    "normocapnia": "Baseline Therapy",
    "normothermia": "Baseline Therapy",
    "first-tier therapy": "Baseline Therapy",

    # --- Therapy: Baseline specifics (now gold classes) ---
    "Analgesia": "Analgesia",
    "analgesia": "Analgesia",
    "analgesic": "Analgesia",
    "Paralysis": "Paralysis",
    "paralysis": "Paralysis",
    "muscle relaxant": "Paralysis",
    "Anti-hypertensives": "Anti-hypertensives",
    "anti-hypertensives": "Anti-hypertensives",
    "antihypertensives": "Anti-hypertensives",
    "Anti-pyretics": "Anti-pyretics",
    "anti-pyretics": "Anti-pyretics",
    "antipyretics": "Anti-pyretics",
    "paracetamol": "Anti-pyretics",
    "Hypothermia": "Hypothermia Therapy",
    "hypothermia": "Hypothermia Therapy",
    "therapeutic hypothermia": "Hypothermia Therapy",
    "induced hypothermia": "Hypothermia Therapy",
    "cooling therapy": "Hypothermia Therapy",
    "CPP-targeted therapy": "Therapy",
    "ICP-targeted therapy": "Therapy",
    "neuroprotective drug": "Therapy",

    # --- Therapy: Secondary insult treatments (now specific gold classes) ---
    "Osmotics (mannitol)": "Osmotics",
    "osmotics": "Osmotics",
    "mannitol": "Osmotics",
    "hyperosmolar therapy": "Osmotics",
    "hypertonic saline": "Osmotics",
    "3% NaCl": "Osmotics",
    "Barbiturates": "Barbiturates",
    "barbiturates": "Barbiturates",
    "thiopental": "Barbiturates",
    "thiopental coma": "Barbiturates",
    "barbiturate coma": "Barbiturates",
    "Steroids": "Steroids",
    "steroids": "Steroids",
    "corticosteroids": "Steroids",
    "dexamethasone": "Steroids",
    "Hyperventilation Therapy": "Secondary Insult Therapy",
    "hyperventilation therapy": "Secondary Insult Therapy",
    "prophylactic hyperventilation": "Secondary Insult Therapy",
    "second-tier therapy": "Secondary Insult Therapy",
    "third-tier therapy": "Secondary Insult Therapy",
    "moderate hypocapnia": "Secondary Insult Therapy",

    # --- Surgical Procedures (now specific gold classes) ---
    "decompressive craniectomy": "Decompressive Craniectomy",
    "hemicraniectomy": "Hemicraniectomy",
    "CSF drainage": "Extra Ventricular Drain Placement",
    "cerebrospinal fluid drainage": "Extra Ventricular Drain Placement",
    "EVD": "Extra Ventricular Drain Placement",
    "external ventricular drain": "Extra Ventricular Drain Placement",
    "ventriculostomy": "Extra Ventricular Drain Placement",
    "lumbar drain": "Extra Ventricular Drain Placement",
    "ICP sensor placement": "ICP Sensor Placement",
    "ICP monitor insertion": "ICP Sensor Placement",
    "ICP bolt": "ICP Sensor Placement",
    "ICP catheter": "ICP Sensor Placement",
    "evacuation of mass lesion": "Evacuation of Mass Lesion",
    "craniotomy": "Evacuation of Mass Lesion",
    "skull fracture elevation": "Skull Fracture Elevation",
    "removal of foreign body": "Removal of Foreign Body",
    "anterior fossa repair": "Anterior Fossa Repair",
    "CSF leak repair": "Anterior Fossa Repair",

    # --- Conditions / insults (all papers) ---
    "raised ICP": "Intracranial Hypertension",
    "raised intracranial pressure": "Intracranial Hypertension",
    "increased intracranial pressure": "Intracranial Hypertension",
    "intracranial hypertension": "Intracranial Hypertension",
    "refractory intracranial hypertension": "Intracranial Hypertension",
    "hypoxia": "Secondary Insult",
    "ischemia": "Condition",
    "cerebral ischemia": "Condition",
    "cerebral hypoperfusion": "Condition",
    "delayed cerebral ischemia": "Condition",
    "vasospasm": "Condition",
    "cerebral edema": "Condition",
    "brain swelling": "Condition",
    "diffuse axonal injury": "Condition",
    "DAI": "Condition",
    "subarachnoid hemorrhage": "Condition",
    "subarachnoid haemorrhage": "Condition",
    "SAH": "Condition",
    "intracerebral hemorrhage": "Condition",
    "intracerebral haemorrhage": "Condition",
    "ICH": "Condition",
    "subdural hematoma": "Condition",
    "subdural haematoma": "Condition",
    "extradural hematoma": "Condition",
    "extradural haematoma": "Condition",
    "hydrocephalus": "Condition",
    "mass lesion": "Condition",
    "impaired cerebral autoregulation": "Cerebrovascular Autoregulation",
    "cerebrovascular autoregulation": "Cerebrovascular Autoregulation",
    "autoregulatory status": "Cerebrovascular Autoregulation",
    "impaired autoregulation": "Cerebrovascular Autoregulation",
    "cerebral autoregulation": "Cerebrovascular Autoregulation",
    "autoregulation": "Cerebrovascular Autoregulation",
    "autoregulatory reserve": "Cerebrovascular Autoregulation",
    "brain death": "Outcome",
    "hypotensive event": "Hypotensive Event",
    "EUSIG-defined hypotensive event": "Hypotensive Event",
    "hypotensive episode": "Hypotensive Event",
    "blood pressure insult": "Hypotensive Event",

    # --- Laboratory values (papers 1, 2, 4) ---
    "paO2": "Blood Gases",
    "PaO2": "Blood Gases",
    "pCO2": "Blood Gases",
    "PaCO2": "Blood Gases",
    "pH": "Blood Gases",
    "arterial blood gas": "Blood Gases",
    "ABG": "Blood Gases",
    "S100B": "Biochemistry",
    "s100b": "Biochemistry",

    # --- Nursing interventions (papers 1, 5) ---
    "endotracheal suction": "Routine Nursing Care",
    "suction": "Routine Nursing Care",
    "eye/mouth care": "Routine Nursing Care",
    "bed care": "Routine Nursing Care",
    "patient turning": "Routine Nursing Care",
    "blood sampling": "Bedside Intervention",
    "transducer calibration": "Bedside Intervention",
    "line insertion": "Bedside Intervention",

    # --- Data quality (papers 2, 5) ---
    "artefact": "Artefact",
    "artifact": "Artefact",
    "physiological artefact": "Artefact",
    "signal artefact": "Artefact",
    "monitoring artefact": "Artefact",

    "guideline adherence": "Guideline Adherence",
    "non-adherence": "Guideline Adherence",
    "protocol adherence": "Guideline Adherence",
    "guideline compliance": "Guideline Adherence",
    "protocol compliance": "Guideline Adherence",
    "BTF guideline": "Guideline Adherence",

    # --- Papers 3-11: SBP/DBP variants ---
    "systolic pressure": "Systolic Blood Pressure (SBP)",
    "SBP": "Systolic Blood Pressure (SBP)",
    "systolic arterial pressure": "Systolic Blood Pressure (SBP)",
    "BPd": "Diastolic Blood Pressure (DBP)",
    "ABPd": "Diastolic Blood Pressure (DBP)",
    "diastolic pressure": "Diastolic Blood Pressure (DBP)",
    "DBP": "Diastolic Blood Pressure (DBP)",
    "pulse pressure": "Pulse Pressure",
    "PP": "Pulse Pressure",

    # --- Papers 3-11: ECG ---
    "electrocardiogram": "Electrocardiogram (ECG)",
    "ECG": "Electrocardiogram (ECG)",
    "EKG": "Electrocardiogram (ECG)",

    # --- Papers 3-11: EUSIG ---
    "EUSIG": "EUSIG Grade",
    "EUSIG grade": "EUSIG Grade",
    "EUSIG classification": "EUSIG Grade",
    "Edinburgh University Secondary Insult Grades": "EUSIG Grade",
    "Edinburgh secondary insult grade": "EUSIG Grade",
    "secondary insult grading": "EUSIG Grade",

    # --- Papers 3-11: TIL ---
    "TIL": "Therapy Intensity Level (TIL)",
    "TIL basic": "Therapy Intensity Level (TIL)",
    "therapy intensity level": "Therapy Intensity Level (TIL)",
    "therapy intensity": "Therapy Intensity Level (TIL)",
    "treatment intensity level": "Therapy Intensity Level (TIL)",

    # --- Papers 3-11: Surgical subtypes ---
    "bifrontal craniectomy": "Bifrontal Craniectomy",
    "bifrontal DC": "Bifrontal Craniectomy",
    "bifrontal decompressive craniectomy": "Bifrontal Craniectomy",
    "cranioplasty": "Cranioplasty",
    "bone flap replacement": "Cranioplasty",
    "skull reconstruction": "Cranioplasty",

    # --- Papers 3-11: Injury mechanism ---
    "injury mechanism": "Injury Mechanism",
    "mechanism of injury": "Injury Mechanism",
    "fall": "Injury Mechanism",
    "road traffic accident": "Injury Mechanism",
    "RTA": "Injury Mechanism",
    "assault": "Injury Mechanism",
    "sport injury": "Injury Mechanism",

    # --- Papers 3-11: Head elevation ---
    "head elevation": "Head Elevation",
    "30-degree head elevation": "Head Elevation",
    "head of bed elevation": "Head Elevation",
    "head-up positioning": "Head Elevation",

    # --- Papers 3-11: Mortality ---
    "mortality": "Mortality",
    "death": "Mortality",
    "in-hospital mortality": "Mortality",
    "ICU mortality": "Mortality",
    "30-day mortality": "Mortality",

    # --- Papers 3-11: Additional therapy synonyms ---
    "pentothal": "Barbiturates",
    "pentobarbital": "Barbiturates",
}


def resolve_class_synonyms(parsed: dict) -> dict:
    """Remap extracted class labels using CLASS_SYNONYM_MAP (canonicalization only)."""
    if not parsed:
        return parsed

    label_lower_map = {k.lower(): v for k, v in CLASS_SYNONYM_MAP.items()}

    def _resolve(label: str) -> str:
        if label in CLASS_SYNONYM_MAP:
            return CLASS_SYNONYM_MAP[label]
        low = label.lower().strip()
        if low in label_lower_map:
            return label_lower_map[low]
        return label

    classes = []
    seen_labels: set[str] = set()
    for c in parsed.get("classes") or []:
        raw_label = (c.get("label") or "").strip()
        if not raw_label:
            continue
        resolved = _resolve(raw_label)
        norm = resolved.lower()
        if norm in seen_labels:
            continue
        seen_labels.add(norm)
        c["label"] = resolved
        classes.append(c)

    relations = []
    for r in parsed.get("relations") or []:
        r["domain"] = _resolve((r.get("domain") or "").strip())
        r["range"] = _resolve((r.get("range") or "").strip())
        relations.append(r)

    hierarchy = []
    for h in parsed.get("hierarchy") or []:
        sub = _resolve((h.get("subClass") or "").strip())
        sup = _resolve((h.get("superClass") or "").strip())
        if sub and sup and sub.lower() != sup.lower():
            h["subClass"] = sub
            h["superClass"] = sup
            hierarchy.append(h)

    return {"classes": classes, "relations": relations, "hierarchy": hierarchy,
            **{k: v for k, v in parsed.items() if k not in ("classes", "relations", "hierarchy")}}


def is_governance_input(text: str) -> bool:
    """True if input is Part A (governance/operations): consortium, core dataset, forms, SQL, ethics, etc."""
    lower = text.lower()
    return any(kw in lower for kw in GOVERNANCE_KEYWORDS)
