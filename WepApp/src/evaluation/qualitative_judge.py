"""LLM-as-judge qualitative evaluation of extracted ontology triples.

Two-stage classification per triple:
    Stage A  →  accept_direct | accept_indirect | reject | revise
    Stage B  →  (only if revise)  4.1–4.10

The rubric is encoded in `STAGE_A_PROMPT` and `STAGE_B_PROMPT`. It mirrors
`docs/qualitative_eval/codebook.md`. Keep the two in sync.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..prompting.llm_client import LLMClient


# ── Rubric constants ─────────────────────────────────────────────────

STAGE_A_LABELS = ("accept_direct", "accept_indirect", "reject", "revise")
STAGE_B_LABELS = (
    "4.1", "4.2", "4.3", "4.4", "4.5",
    "4.6", "4.7", "4.8", "4.9", "4.10",
)

# Short label descriptions — rendered inline in prompts and shown in the UI.
STAGE_A_DESCRIPTIONS = {
    "accept_direct": (
        "The triple is semantically correct and holds directly between the "
        "two named classes at their current level of specificity."
    ),
    "accept_indirect": (
        "The triple is semantically correct, BUT the relationship does not "
        "hold directly between the two named classes — it is only true via "
        "a traceable chain of intermediate concepts. You must be able to "
        "spell out the intermediate steps. Example: "
        "'DataQualityAssessment wasAssociatedWith Condition' is accept_indirect "
        "because the path is: DQA → assesses → Observation → measuresParameter "
        "→ Patient → hasCondition → Condition. The triple IS true, but only "
        "through that chain. Contrast with accept_direct where a single arrow "
        "connects the two classes (e.g. 'Outcome influencedBy Intervention'). "
        "A single child→parent step is accept_direct, NOT accept_indirect."
    ),
    "reject": (
        "The triple is semantically incorrect or clinically implausible — "
        "the stated relationship does not hold (directly or indirectly) "
        "between the two named classes, AND no small edit (swapping, "
        "renaming, or re-targeting) would fix it. Examples: "
        "'Disability isInfluencedBy GcsAssessment'; "
        "'GuidelineAdherence receivesTherapy TraumaticBrainInjury'. "
        "If a swap, rename, or target change WOULD fix it, choose REVISE "
        "instead."
    ),
    "revise": (
        "The triple is close to correct but needs a specific modelling "
        "correction — such as changing the target class, fixing the "
        "relation label, swapping subject/object, converting between "
        "subclass and direct relationship, or fixing a class name."
    ),
}

STAGE_B_DESCRIPTIONS = {
    "4.1": "Change target — more general (move UP the hierarchy).",
    "4.2": "Change target — more specific (move DOWN the hierarchy).",
    "4.3": "Change target — different class entirely (no hierarchy move).",
    "4.4": "Delete and add a new direct relationship (the correct relation holds directly between the two classes, e.g. 'isMeasuredBy' → 'assesses').",
    "4.5": "Delete and add a new indirect relationship (the correct relationship requires intermediate concepts, e.g. 'GuidelineAdherence receivesTherapy TBI' → 'GuidelineAdherence forInsultTreatment TBI' because the true path is GuidelineAdherence → Guideline → forTreating → TBI).",
    "4.6": "Edit class name (plural/singular, casing, typo).",
    "4.7": "Delete subclass edge, replace with a direct relationship.",
    "4.8": "Swap subject and object.",
    "4.9": "Change relationship to a subclass edge.",
    "4.10": "Other revise action with a concrete suggested edit.",
}


# ── Prompt templates ────────────────────────────────────────────────

def _format_class_block(label: str,
                        class_lookup: Dict[str, Dict[str, Any]],
                        max_def: int = 240) -> str:
    info = class_lookup.get(label) or {}
    defn = (info.get("definition") or "").strip()
    if not defn:
        return f"{label}"
    defn = defn.replace("\n", " ").strip()
    if len(defn) > max_def:
        defn = defn[: max_def - 1].rstrip() + "…"
    return f"{label} — {defn}"


def build_stage_a_prompt(triple: Dict[str, Any],
                         class_lookup: Dict[str, Dict[str, Any]]) -> str:
    subject = triple["subject"]
    relation = triple["relation"]
    obj = triple["object"]
    evidence = (triple.get("evidence") or "").strip()
    kind = triple.get("kind", "relation")

    evidence_block = evidence if evidence else "(none provided)"
    evidence_status = "text-grounded" if evidence else "missing"

    subj_def = _format_class_block(subject, class_lookup)
    obj_def = _format_class_block(obj, class_lookup)

    rubric = "\n".join(
        f"- {k}: {v}" for k, v in STAGE_A_DESCRIPTIONS.items()
    )

    return f"""You are a senior ontology engineer AND a neurointensive-care domain expert reviewing a single triple extracted by an LLM from clinical literature. Decide what action a knowledge engineer would take.

DOMAIN CONTEXT — BrainIT Ontology Conventions:
This ontology models the BrainIT neurointensive-care database for TBI patients.
The lists below are EXAMPLES of valid relationships — they are NOT exhaustive.
A subclass relationship can be valid even if not listed here. Judge each triple
on its own clinical and ontological merit.

Key modelling conventions:
• "Therapy" is the top-level class for all treatments. It has subclasses
  including "Baseline Therapy" and "Secondary Insult Therapy", among others.
  "Patient Interventions" is also a valid subclass of Therapy.
• "Baseline Therapy" = standard ICU therapies. Valid subclasses include (among
  others): Sedation, Fluids, Nutrition, Ventilation, Analgesia, Antibiotics,
  Anti-pyretics, Anti-hypertensives, Hypothermia Therapy, Paralysis (NMB).
• "Secondary Insult Therapy" = targeted therapies. Valid subclasses include
  (among others): Barbiturates, Osmotics, Steroids, Arterial Pressors.
• "Monitoring Data" is the top-level class for all monitored values. It has
  subclasses including "Core Monitoring Parameter", "Optional Monitoring
  Parameter", and "Derived Parameter", among others.
• "Core Monitoring Parameter" subclasses include (among others): ICP, MAP,
  CPP, Heart Rate, Respiration Rate, Temperature, SaO2, CVP.
• "Optional Monitoring Parameter" subclasses include (among others): TCD,
  SjO2, EtCO2, NIBP, PbrO2, Brain Temperature, Cardiac Output, Peripheral
  Temperature, Microdialysis.
• "Derived Parameter" subclasses include (among others): CPP, PRx.
• "Physiological Data" is a valid class. Specific parameters (Heart Rate,
  ICP, MAP, CPP, Carbon Dioxide, PbrO2, etc.) can validly be subclasses
  of BOTH their monitoring category AND Physiological Data.
• "Condition" is a GENERIC superclass for ALL medical conditions — NOT
  limited to stroke. Valid subclasses include (among others): TBI, Sepsis,
  Hypotension, Hypertension, Adverse Event, Brain Injury, etc.
• "Brain Injury" is a valid class. TBI, Secondary Brain Injury, Head Injury,
  Neurological Injury, and Coma are valid subclasses of Brain Injury.
• "Secondary Insult" = secondary complications. Valid subclasses include
  (among others): Hypotension, Hypertension, Intracranial Hypertension,
  Arterial Hypotension, Systemic Hypotension, ARDS, Sepsis, Jugular Venous
  Desaturation, Cerebrovascular Autoregulation impairment.
• "Laboratory Values" has sub-panels: Biochemistry, Haematology, Blood Gases.
  Individual analytes (Sodium, Potassium, Glucose, Haemoglobin, Haematocrit,
  WBC) are valid subclasses of their parent panel.
• "Clinical Assessment" includes (among others): GCS Assessment, CT Scan
  Assessment, Pupil Assessment.
• "Nursing Intervention" includes (among others): Physiotherapy, Patient
  Transport, Routine Nursing Care, Bedside Intervention.
• "Surgical Procedure" includes (among others): Decompressive Craniectomy,
  EVD Placement, ICP Sensor Placement, Anterior Fossa Repair, Skull Fracture
  Elevation, Evacuation of Mass Lesion, Removal of Foreign Body.
• "Outcome" includes (among others): Mortality, GOSe Outcome, Outcome Score.
• "Clinical Practice" includes (among others): Guideline Adherence, Clinical
  Reminders.
• "Observation" has subclasses including Reading and Parameter. Machine is
  a valid subclass of Sensor.
• "Data Quality Assessment" includes (among others): Possible Error,
  Probable Error.
• Electrocardiogram (ECG) is a valid subclass of Monitoring Data.

Additional valid hierarchy relationships from SSN/BrainIT alignment:
• Parameter subClassOf Observation (SSN alignment: pd:Parameter subClassOf ssn:Property).
• Reading subClassOf Observation (SSN alignment: pd:Reading subClassOf ssn:ObservationValue).
• Machine subClassOf Sensor (SSN alignment: mms:Machine subClassOf ssn:Platform).
• Secondary Insult subClassOf Insult.
• Guideline Adherence subClassOf Clinical Practice.
• Clinical Reminders subClassOf Clinical Practice.

Additional BrainIT expert-model conventions (from the authoritative BrainIT
ontology diagrams — these are AUTHORITATIVE and supersede any auto-extracted
definition):
• "Reading" is the hub class for all physiological observations. Every
  specific physiological parameter is a subclass of Reading, usually via
  a body-system intermediary:
  - "Cardiac Reading" subclasses: Heart Rate, "Cardiac Pressure Reading"
    → "Blood Pressure Reading" (BPs, BPd, BPm), "Central Venous Pressure
    Reading" (CVPs, CVPm, CVPd), Cardiac Output, Cardiac Input.
  - "Brain Reading" subclasses: "Brain Tissue Reading" (Brain Temp),
    "Brain Pressure Reading" (ICPs, ICPd, ICPm), "Calculated Brain
    Pressure Reading" (CPP), "Compliance Reading" (PVI, Comp).
  - "Respiratory Reading" subclasses: Respiratory Rate.
  These body-system groupings COEXIST with the "Core/Optional/Derived
  Monitoring Parameter" taxonomy — both classifications are valid for the
  same parameter.
• A "Physiological Reading" is a REIFIED n-ary node with (exactly 1):
  Patient, Variable, Value, Timestamp; plus Source → Zeroing Location,
  Source Type, Source Location (Measurement Site, Measurement Side,
  Depth). Valid structural relations: hasPatient, forVariable, hasValue,
  hasTimestamp, hasSource, hasZeroingLocation, hasSourceType,
  hasSourceLocation, hasMeasurementSite, hasMeasurementSide, hasDepth,
  hasProximityToNeuropathology. A flat triple inevitably captures only
  ONE facet of this reified node — do NOT reject a triple merely because
  it is structurally incomplete; assess the semantic truth of the flat
  edge only.
• "Value" is itself a class with hasUnitOfMeasurement → "Unit of
  Measurement" (mmHg, mls) and a numeric readingValue.
• "Daily Observations" subClassOf Reading. Valid subclasses: Daily
  Sedation, Daily Analgesia, Daily Paralysis, Daily Antibiotic, Daily
  Nutrition, Daily Vasopressor, Daily Fluid (→ Daily Fluid Input, Daily
  Fluid Output). Each Daily Observation connects to "Delivery Mechanism"
  (Bolus, Infusion, Enteral, Parenteral, Both, None) via
  `deliveryMechanism` / `hasDeliveryMechanism` / `isAdministeredVia` —
  all three relation labels are valid.
• "Demographic" is a top-level class (Patient id, BrainIT Center).
  Subclasses include: Sex (Male, Female), Age, "Trauma Event" (with
  "Trauma Event Type": Traffic Accident, Fall, Pedestrian, Work, Assault,
  Sport, Unknown), "Previous Dysfunction" (No, Mental, Head Trauma, Head
  Trauma and Mental Dysfunction, Unknown), "Influence of Alcohol",
  "Injury" (→ Haemorrhage, Chest, Spinal, Pelvic, Abdominal, Facial,
  Limb, Other). Valid relations: hasExistence (→ "Existence": Yes,
  Suspected, No, Unknown, Confirmed), hasSpecifics, hasA.
• The authoritative BrainIT ontology also uses SPACED relation labels
  (e.g. "has Patient", "for Variable", "has Unit Of Measurement",
  "has Reading"). These are canonical BrainIT relation names and must
  NOT be rejected as "natural language" — treat them as equivalent to
  their camelCase forms.

Valid BrainIT RELATION labels include (among others): isAdministeredDuring,
isMonitoredBy, hasMonitoringData, hasOutcome, targetsCondition, receivesTherapy,
hasClinicalAssessment, hasLaboratoryValue, hasNursingIntervention,
hasSurgicalProcedure, isAssociatedWith, isInfluencedBy, affectsOutcome,
isARiskFactorFor, defines, informs, worsens, reduces, assesses,
hasQualityAssessment, hasDemographicData, producedBySensor, measuresParameter,
isAConditionOf, hasTimepoint, isRelatedTo, calculatesFrom, correlatesWith.
Do NOT reject a triple just because the relation label is unfamiliar — if the
relation name is a reasonable domain predicate, accept or revise it.

CRITICAL: The lists above are NON-EXHAUSTIVE examples. Do NOT reject a triple
solely because a class or relation is not mentioned here. If the subclass
relationship or association is clinically reasonable in neurointensive care,
it is likely valid.

TRIPLE TYPE: {kind}

TRIPLE
  subject:  {subj_def}
  relation: {relation}
  object:   {obj_def}

IMPORTANT — The class definitions above were auto-extracted by an LLM and
are FREQUENTLY WRONG (e.g. "Condition" may be defined as "a type of stroke"
when the label clearly means any medical condition; "Insult" may describe
a narrow event when the label covers the full BrainIT class). You MUST
IGNORE any class definition that contradicts the DOMAIN CONTEXT preamble.
The preamble is authoritative; the definitions are supplementary and often
noisy. NEVER reject a triple because the auto-extracted definition seems
to contradict it — the preamble wins, always. When in doubt between the
preamble and the definition, believe the preamble.

EVIDENCE (verbatim from source text, if any)
  status: {evidence_status}
  text:   {evidence_block}

RUBRIC — pick EXACTLY ONE of:
{rubric}

DECISION PROCESS — follow these steps IN ORDER:

Step 0 — Preamble precedence (hierarchy triples only):
  If the triple's relation is `subClassOf` AND the DOMAIN CONTEXT preamble
  explicitly lists the subject as a valid subclass of the object (either
  directly or transitively via the preamble's bullet lists), the verdict
  is accept_direct IMMEDIATELY. Do NOT consult the auto-extracted class
  definitions — they may be noisy. The preamble is authoritative.
  Examples that MUST be accept_direct under Step 0:
  • "TBI subClassOf Condition" (Condition is generic; TBI is listed)
  • "Hypotension subClassOf Condition" / "Hypertension subClassOf Condition"
  • "Sepsis subClassOf Condition" / "Adverse Event subClassOf Condition"
  • "Brain Injury subClassOf Condition"
  • "Paralysis subClassOf Baseline Therapy" (Paralysis/NMB is listed)
  • "ARDS subClassOf Secondary Insult" (ARDS is listed)
  • "Cerebrovascular Autoregulation subClassOf Secondary Insult"
  • "Heart Rate subClassOf Cardiac Reading" / "ICPs subClassOf Brain
    Pressure Reading" / "CPP subClassOf Calculated Brain Pressure
    Reading" / "BPs subClassOf Blood Pressure Reading" / "CVPd subClassOf
    Central Venous Pressure Reading" (body-system Reading tree).
  • "Daily Sedation subClassOf Daily Observations" / "Daily Fluid Input
    subClassOf Daily Fluid" (Daily Observations tree).
  • "Traffic Accident subClassOf Trauma Event Type" / "Haemorrhage
    subClassOf Injury" / "Confirmed subClassOf Existence" /
    "Male subClassOf Sex" (Demographic tree).
  • Any analyte subClassOf its parent panel, any listed Nursing
    Intervention child subClassOf Nursing Intervention, any listed
    Baseline/Secondary Insult Therapy drug subClassOf its therapy class.
  → If Step 0 matches: verdict = accept_direct. STOP.

Step 1: Does the triple hold directly between the two named classes?
  → YES: verdict = accept_direct. STOP.

Step 2: Does the triple hold indirectly — via a traceable chain of
  intermediate concepts you can name?
  → YES: verdict = accept_indirect. STOP.

Step 3 (MANDATORY before reject — you MUST mentally construct and evaluate
EACH candidate below before deciding; your justification must show you did):

  Candidate A — SWAP: Form "<object> <relation> <subject>".
    Is THAT swapped triple correct ON ITS OWN? (Evaluate the swap, not the
    original.) Only pick revise via swap if the swapped triple is valid.
    GOOD swap: "Mortality isARiskFactorFor TBI" →
      "TBI isARiskFactorFor Mortality" ✓ (TBI really is a risk factor for mortality).
    BAD swap: "Analgesia reduces Poor Outcome" →
      "Poor Outcome reduces Analgesia" ✗ (an outcome cannot reduce a therapy;
      do NOT pick swap here — try Candidate B or D instead).
    → If swapped triple IS correct: verdict = revise.

  Candidate B — RELABEL: Replace the relation with the single closest
    valid BrainIT predicate from the DOMAIN CONTEXT whitelist. Does
    "<subject> <new_relation> <object>" become correct?
    → If yes: verdict = revise.

  Candidate C — HIERARCHY-CONVERT: Does the DOMAIN CONTEXT preamble
    classify the subject as a subclass (direct or transitive) of the
    object, or vice versa? This applies EVEN IF the current relation
    label is not "isATypeOf" — the preamble lists analyte→panel,
    specific-nursing-intervention→Nursing Intervention, specific-drug→
    therapy-class, etc. as valid subclass facts.
    Examples:
      • "Sodium hasLaboratoryValue Biochemistry" — preamble says analytes
        are subclasses of their panel → fix is Sodium subClassOf Biochemistry.
      • "Bedside Intervention hasNursingIntervention NursingIntervention"
        — preamble lists Bedside Intervention as a valid subclass →
        fix is subClassOf.
    → If the preamble implies subclass: verdict = revise.

  Candidate D — ENDPOINT-REPLACE: Would replacing ONE endpoint with a
    more general, more specific, or sibling class make it correct?
    Example: "Intracranial Hypertension receivesTherapy GOSe Outcome" —
    object should be replaced with an actual therapy.
    → If yes: verdict = revise.

  Candidate E — NAME-FIX: Purely a lexical issue (plural→singular,
    casing, typo)?
    → If yes: verdict = revise.

Step 3b — ROLE-TYPE check (prevents false rejects on type mismatches):
  Some relations impose type constraints on their endpoints. A TYPE
  MISMATCH is almost always REVISE, not reject — the author intended a
  valid structure and picked the wrong piece.
  • "receivesTherapy" — object MUST be a Therapy. A parameter, outcome,
    or other condition in object position → fix is Candidate D (replace
    object with a real therapy) or Candidate B (relabel).
  • "isMonitoredBy" — object MUST be a sensor/assessment/monitoring method.
  • "hasOutcome" / "affectsOutcome" — object MUST be an Outcome.
  • "hasLaboratoryValue" — object MUST be an analyte; if subject is an
    analyte and object is a panel, the fact is subclass-shaped (Candidate C).
  • "hasClinicalAssessment" — object MUST be a Clinical Assessment.

Step 4: ONLY if EVERY candidate in Step 3 AND the Step 3b role-type check
  fails → verdict = reject. Your justification MUST name which candidates
  you considered and why each one failed (e.g. "swap gives nonsense; no
  whitelist relation fits; endpoints are not hierarchically related in the
  preamble; no endpoint replacement produces a valid triple").

Tie-breaker: When uncertain between reject and revise, CHOOSE REVISE. A
wasted Stage B call is cheap; a silent reject of a salvageable triple is
irreversible in the pipeline.

Additional rules:
- Apply CONSISTENT verdicts across structurally similar triples (e.g. if
  one analyte→panel pairing is 4.9, ALL analyte→panel pairings are 4.9;
  if one Core/Optional Monitoring Parameter with `hasMonitoringData
  MonitoringData` is 4.9, ALL such pairings are 4.9 — do NOT split
  identical patterns between 4.4 and 4.9).
- Cross-triple consistency: any triple of the form "<Core or Optional
  Monitoring Parameter> hasMonitoringData MonitoringData" (e.g. HR, ICP,
  MAP, CPP, Respiration Rate, NIBP, SaO2, SjO2, TCD, EtCO2, PbrO2, Brain
  Temperature) is always 4.9 (subClassOf). These parameters ARE monitoring
  data, so the edge is hierarchical, not relational.
- When evidence is missing, confidence MUST be ≤ 0.70 and flags MUST
  include "low_evidence".
- Use "out_of_scope" flag when the triple leaves the neuro-ICU / BrainIT
  domain.

Respond with ONLY a single JSON object, no prose, no code fences. The
"justification" field MUST name the SPECIFIC classes/relation involved
and the concrete reason (e.g. "Sodium→Biochemistry is the analyte→panel
subclass pattern listed in the preamble; relation should be subClassOf").
Generic filler like "the relation label needs to be changed" is NOT
acceptable:
{{"verdict": "<one of accept_direct|accept_indirect|reject|revise>", "justification": "<one sentence naming specific classes/relation, ≤40 words>", "confidence": <float 0.0–1.0>, "flags": [<zero or more of "low_evidence","ambiguous_domain","out_of_scope">]}}
"""


def build_stage_b_prompt(triple: Dict[str, Any],
                         class_lookup: Dict[str, Dict[str, Any]],
                         stage_a_justification: str) -> str:
    subject = triple["subject"]
    relation = triple["relation"]
    obj = triple["object"]
    kind = triple.get("kind", "relation")

    subj_def = _format_class_block(subject, class_lookup)
    obj_def = _format_class_block(obj, class_lookup)

    rubric = "\n".join(
        f"- {k}: {v}" for k, v in STAGE_B_DESCRIPTIONS.items()
    )

    return f"""The triple below was classified as REVISE. Identify the SPECIFIC revise action a knowledge engineer would take and propose the corrected triple.

DOMAIN CONTEXT — BrainIT Ontology Conventions:
This ontology models the BrainIT neurointensive-care database for TBI patients.
Key conventions:
• "Baseline Therapy" = standard ICU therapies (Sedation, Fluids, Nutrition,
  Ventilation, Analgesia, Antibiotics, Anti-pyretics, Anti-hypertensives,
  Hypothermia Therapy, Paralysis/NMB).
• "Secondary Insult Therapy" = targeted therapies (Barbiturates, Osmotics,
  Steroids, Arterial Pressors).
• "Condition" is a GENERIC superclass for ALL medical conditions — NOT stroke.
• Specific analytes (e.g. Haemoglobin) being subClassOf their panel (e.g.
  Haematology) is VALID in BrainIT — do NOT change it.
• Expert-model subtrees (from the authoritative BrainIT ontology):
  - "Reading" is the hub for physiological observations. Body-system
    subtrees: "Cardiac Reading" (Heart Rate, Blood Pressure Reading
    BPs/BPd/BPm, Central Venous Pressure Reading CVPs/CVPm/CVPd, Cardiac
    Output/Input); "Brain Reading" (Brain Tissue Reading → Brain Temp;
    Brain Pressure Reading → ICPs/ICPd/ICPm; Calculated Brain Pressure
    Reading → CPP; Compliance Reading → PVI/Comp); "Respiratory Reading"
    → Respiratory Rate. These subclass facts are CORRECT and must be
    preserved.
  - "Daily Observations" subClassOf Reading. Subclasses: Daily Sedation,
    Daily Analgesia, Daily Paralysis, Daily Antibiotic, Daily Nutrition,
    Daily Vasopressor, Daily Fluid (→ Input/Output). Each connects to
    "Delivery Mechanism" (Bolus, Infusion, Enteral, Parenteral, Both,
    None) via `deliveryMechanism` / `hasDeliveryMechanism` /
    `isAdministeredVia` — all valid.
  - "Demographic" subclasses: Sex (Male/Female), Age, Trauma Event (with
    Trauma Event Type: Traffic Accident/Fall/Pedestrian/Work/Assault/
    Sport), Previous Dysfunction, Influence of Alcohol, Injury
    (Haemorrhage/Chest/Spinal/Pelvic/Abdominal/Facial/Limb/Other).
    Relations: hasExistence → "Existence" (Yes/Suspected/No/Unknown/
    Confirmed), hasSpecifics, hasA — all valid.
• Spaced relation labels ("has Patient", "for Variable", "has Unit Of
  Measurement", "has Reading") are CANONICAL BrainIT names — treat them
  as equivalent to their camelCase forms; do NOT reject as natural language.
• Valid BrainIT relation labels include (among others): isAdministeredDuring,
  isMonitoredBy, hasMonitoringData, hasOutcome, targetsCondition,
  receivesTherapy, hasClinicalAssessment, hasLaboratoryValue,
  hasNursingIntervention, hasSurgicalProcedure, isAssociatedWith,
  isInfluencedBy, affectsOutcome, hasQualityAssessment, defines, informs,
  hasPatient, forVariable, hasValue, hasUnitOfMeasurement, hasSource,
  hasMeasurementSite, deliveryMechanism, hasExistence, hasSpecifics.
  Do NOT change these to natural-language alternatives.

TRIPLE TYPE: {kind}

TRIPLE
  subject:  {subj_def}
  relation: {relation}
  object:   {obj_def}

IMPORTANT — Class definitions above were auto-extracted and are
FREQUENTLY WRONG. IGNORE any class definition that contradicts the
DOMAIN CONTEXT preamble — the preamble wins. Do NOT change a correct
triple into something else because an auto-extracted definition looks
noisy (e.g. do NOT rewrite "Barbiturates subClassOf Secondary Insult
Therapy" into anything else — the preamble explicitly places Barbiturates
under Secondary Insult Therapy). When writing the suggested_triple, use
ONLY the short class label (e.g. "Carbon Dioxide"), NEVER include class
definitions or comments in the name.

STAGE-A REASONING: {stage_a_justification}

SUB-TYPES — pick EXACTLY ONE:
{rubric}

Order of check (take the FIRST that applies):
1) Label lexical fix only (plural/case/typo)       → 4.6
   Example: "BrainInjuries" → "BrainInjury" (just a plural fix).
2) Subject and object are swapped                   → 4.8
   CRITICAL: If the triple becomes correct by exchanging subject and object
   (keeping the same relation), the answer is ALWAYS 4.8 — not 4.4.
   Examples of 4.8:
   • "DemographicData hasDemographicData Patient" → swap to
     "Patient hasDemographicData DemographicData"
   • "Mortality isARiskFactorFor TBI" → swap to
     "TBI isARiskFactorFor Mortality"
   • "Sensor producedBySensor Reading" → swap to
     "Reading producedBySensor Sensor"
   If your proposed fix changes the subject to the original object (or vice
   versa) while keeping the relation meaning, that IS a swap → 4.8.
   CORRECTNESS GATE: Before committing to 4.8, construct the swapped triple
   and verify it is VALID on its own. If the swap is ALSO nonsensical (e.g.
   "Poor Outcome reduces Analgesia" — an outcome cannot reduce a therapy),
   do NOT pick 4.8 — try 4.4 (relabel) or 4.3 (replace endpoint) instead.
3) A subclass/hierarchy fact expressed as a relation → 4.9
   HARD RULE: If the input relation string contains any of `isATypeOf`,
   `isA`, `typeOf`, `is a type of`, `is a kind of` (case-insensitive),
   the sub-type is ALWAYS 4.9 — even if the endpoints seem questionable.
   Emit the suggested triple with `subClassOf`. If the endpoints are
   genuinely wrong, combine with an endpoint replacement in the
   suggested_triple (still 4.9, not 4.3).
   OTHERWISE, 4.9 triggers when EITHER:
   (a) the relation label is `isATypeOf` / `isA` / `typeOf` / variant
       (already covered by the hard rule above); OR
   (b) the DOMAIN CONTEXT preamble classifies the subject as a subclass
       (direct or transitive) of the object. Look at the preamble bullet
       lists — any analyte with its parent panel (Sodium + Biochemistry,
       Haemoglobin + Haematology, …); any listed Nursing Intervention child
       (Bedside Intervention, Physiotherapy, …) with "Nursing Intervention";
       any listed Baseline / Secondary Insult Therapy drug with its therapy
       class; any Core/Optional Monitoring Parameter (ICP, MAP, CPP, HR,
       NIBP, SjO2, TCD, EtCO2, PbrO2, …) with "Monitoring Data"; any
       physiological parameter with its body-system Reading (HR → Cardiac
       Reading; ICPs/ICPd/ICPm → Brain Pressure Reading; CPP → Calculated
       Brain Pressure Reading; BPs/BPd/BPm → Blood Pressure Reading;
       CVPs/CVPm/CVPd → Central Venous Pressure Reading; Brain Temp →
       Brain Tissue Reading; Respiratory Rate → Respiratory Reading); any
       Daily Observation (Daily Sedation/Analgesia/Paralysis/Antibiotic/
       Nutrition/Vasopressor/Fluid) with "Daily Observations"; any
       demographic subclass (Male/Female → Sex; Traffic Accident/Fall/…
       → Trauma Event Type; Haemorrhage/Chest/Spinal/… → Injury;
       Yes/Suspected/No/Unknown/Confirmed → Existence).
   If (b) holds, IGNORE the current relation label — the intended fact is
   a subClassOf edge. Examples:
   • "Sodium hasLaboratoryValue Biochemistry" → 4.9 →
     "Sodium subClassOf Biochemistry" (analyte→panel, per preamble).
   • "Bedside Intervention hasNursingIntervention Nursing Intervention"
     → 4.9 → "Bedside Intervention subClassOf Nursing Intervention".
   • "NIBP hasMonitoringData Monitoring Data" → 4.9 →
     "NIBP subClassOf Monitoring Data" (Optional Monitoring Parameter).
   • "Heart Rate hasMonitoringData Monitoring Data" → 4.9 →
     "Heart Rate subClassOf Monitoring Data" (Core Monitoring Parameter).
   • "CPP hasReading Calculated Brain Pressure Reading" → 4.9 →
     "CPP subClassOf Calculated Brain Pressure Reading" (body-system tree).
   • "Daily Sedation hasObservation Daily Observations" → 4.9 →
     "Daily Sedation subClassOf Daily Observations" (Daily tree).
   • "ExtracranialInjury isATypeOf Injuries" → 4.9 →
     "ExtracranialInjury subClassOf Injury".
   Suggested_triple MUST use `subClassOf` — never `isATypeOf` in the output.
4) Subclass/hierarchy used, but should be a relation → 4.7
   Example: "GuidelineAdherence subClassOf Guideline" → should be
   "GuidelineAdherence isAssociatedWith Guideline".
5) Endpoints correctly paired, but relation label is wrong → 4.4 (default) or 4.5 (rare)
   DEFAULT to 4.4: the corrected relation name connects the two endpoints
   DIRECTLY via a single BrainIT predicate from the whitelist. Examples:
   `hasMonitoringData` → `isMonitoredBy`; `triggersIntervention` →
   `isARiskFactorFor`; `hasClinicalAssessment` → `hasOutcome`;
   `isTreatedWith` → `receivesTherapy`. The overwhelming majority (~90%)
   of relation-label fixes are 4.4.
   ONLY pick 4.5 when the corrected relation label ITSELF implies
   transitivity / chain traversal AND no direct whitelist predicate
   connects the two classes. If ANY direct whitelist relation fits
   (hasOutcome, isMonitoredBy, receivesTherapy, affectsOutcome,
   isARiskFactorFor, assesses, hasLaboratoryValue, isAssociatedWith,
   isInfluencedBy, targetsCondition, hasNursingIntervention,
   hasSurgicalProcedure, isAdministeredDuring, …), the answer is 4.4.
   4.5 is rare and reserved for cases like: "GuidelineAdherence
   receivesTherapy TBI" → "GuidelineAdherence forInsultTreatment TBI"
   where the fix predicate itself encodes a chain (GuidelineAdherence →
   Guideline → forTreating → TBI).
6) One endpoint is wrong:
     - More general class fixes it  → 4.1
     - More specific class fixes it → 4.2
     - Entirely different class fixes it → 4.3
7) None of the above → 4.10 (MUST give a concrete suggested edit)

Hard constraints on the suggested_triple:
- NEVER invent new classes. Use ONLY class labels that already appear in the
  input triple or are well-known parent/child classes in the BrainIT domain
  (as listed in the domain context above). Classes like "Neurological
  recovery outcome" or "Patient outcome measure" do NOT exist — do not
  create them.
- Use ONLY the short class label in suggested_triple fields. NEVER append
  definitions, comments, or descriptions. Write "Carbon Dioxide", NOT
  "Carbon Dioxide — A gas produced by the body during metabolism".
- Prefer the SMALLEST edit that fixes the issue — change ONE field when
  possible. If swapping fixes it, swap (4.8); do not also rename the
  relation.
- Relation labels MUST be short, camelCase ontology identifiers. Valid
  examples: hasOutcome, isTreatedWith, isMonitoredBy, subClassOf,
  isAssociatedWith, targetsCondition, receivesTherapy, isInfluencedBy,
  affectsOutcome, reduces, worsens, isARiskFactorFor, assesses,
  isAdministeredDuring, hasClinicalAssessment, isDerivedFrom.
  INVALID examples (natural language — do NOT use): "is a complication of",
  "has patient outcome measure", "is performed during", "is recorded by",
  "has specific laboratory value", "is utilized during".
- NEVER produce a self-loop: suggested_triple.subject MUST NOT equal
  suggested_triple.object. If the only "fix" you can construct is a
  self-loop, the original was already correct — return 4.10 with the
  original triple unchanged and explain that no revision was needed.
- suggested_triple MUST mention at least ONE class from the original
  triple (either the original subject or the original object). Replacing
  BOTH endpoints with entirely different classes is a hallucination, not
  a revision — if you cannot fix the triple while keeping at least one
  endpoint, return 4.10 with the original triple unchanged.
- If the ORIGINAL triple is already correct as-is (e.g. it is already
  `subClassOf` with a valid analyte→panel / parameter→Monitoring Data /
  therapy→therapy-class pattern per the preamble), do NOT propose a
  spurious revision — this case should not have reached Stage B. Return
  4.10 with the original triple unchanged and state explicitly that the
  input is already valid so the Stage A verdict should have been accept.
- If no reasonable revision exists, return revise_subtype "4.10", repeat
  the ORIGINAL triple unchanged in suggested_triple, and explain why.

Respond with ONLY a single JSON object, no prose, no code fences. The
"justification" field MUST name the SPECIFIC classes/relation being
corrected and state why (e.g. "Sodium→Biochemistry is the analyte→panel
subclass pattern in the preamble, so relation should be subClassOf").
Generic filler like "the relation label needs to be changed" is NOT
acceptable:
{{"revise_subtype": "<one of 4.1|4.2|4.3|4.4|4.5|4.6|4.7|4.8|4.9|4.10>", "suggested_triple": {{"subject": "...", "relation": "...", "object": "..."}}, "justification": "<one sentence naming specific classes/relation, ≤40 words>", "confidence": <float 0.0–1.0>}}
"""


# ── JSON parsing ─────────────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_strict(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from an LLM response.

    Tolerates code fences and leading/trailing prose. Returns None on failure.
    """
    if not text:
        return None
    candidate = text.strip()
    # Strip common fences first.
    m = _JSON_FENCE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    # If still not a clean object, grab the first {...} block.
    if not candidate.startswith("{"):
        m2 = _FIRST_OBJECT.search(candidate)
        if m2:
            candidate = m2.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_stage_a(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise Stage A output; fallback to 'reject' with low confidence on parse failure."""
    if not isinstance(obj, dict):
        return {
            "verdict": "reject",
            "justification": "judge output unparseable; defaulted to reject",
            "confidence": 0.0,
            "flags": ["parse_error"],
        }
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in STAGE_A_LABELS:
        # Attempt loose matching for common paraphrases.
        if "direct" in verdict and "indirect" not in verdict:
            verdict = "accept_direct"
        elif "indirect" in verdict:
            verdict = "accept_indirect"
        elif "revise" in verdict or "edit" in verdict:
            verdict = "revise"
        elif "reject" in verdict or "incorrect" in verdict:
            verdict = "reject"
        else:
            verdict = "reject"
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    flags = obj.get("flags") or []
    if not isinstance(flags, list):
        flags = []
    if "low_evidence" in flags:
        conf = min(conf, 0.7)
    return {
        "verdict": verdict,
        "justification": str(obj.get("justification") or "").strip(),
        "confidence": conf,
        "flags": [str(f) for f in flags if str(f).strip()],
    }


def _coerce_stage_b(obj: Optional[Dict[str, Any]],
                    fallback_triple: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "revise_subtype": "4.10",
            "suggested_triple": None,
            "justification": "judge output unparseable",
            "confidence": 0.0,
        }
    sub = str(obj.get("revise_subtype", "")).strip()
    if sub not in STAGE_B_LABELS:
        sub = "4.10"
    suggested = obj.get("suggested_triple")
    if isinstance(suggested, dict):
        suggested = {
            "subject": str(suggested.get("subject") or "").strip(),
            "relation": str(suggested.get("relation") or "").strip(),
            "object": str(suggested.get("object") or "").strip(),
        }
        if not (suggested["subject"] and suggested["relation"] and suggested["object"]):
            suggested = None
    else:
        suggested = None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "revise_subtype": sub,
        "suggested_triple": suggested,
        "justification": str(obj.get("justification") or "").strip(),
        "confidence": max(0.0, min(1.0, conf)),
    }


def _sanitize_stage_b(verdict_record: Dict[str, Any]) -> None:
    """Catch and fix common LLM-judge bugs in the suggested_triple.

    Three patterns produce unusable suggestions that leak past the rubric:
      1. Self-loops (subject == object in the proposed triple).
      2. Non-4.10 revise verdicts whose suggested_triple is byte-identical
         to the original — Stage A over-called revise; the correct verdict
         is accept_direct.
      3. Wholesale class substitution — the suggested triple mentions
         neither the original subject nor the original object, indicating
         the judge hallucinated a different pair of classes from the preamble.

    Mutates ``verdict_record`` in place. No-op when verdict != "revise"
    or no suggested_triple is present.
    """
    if verdict_record.get("verdict") != "revise":
        return
    suggested = verdict_record.get("suggested_triple")
    if not isinstance(suggested, dict):
        return

    _paren = re.compile(r"\s*\([^)]*\)\s*")

    def _norm(s: Any) -> str:
        s = str(s or "").strip().lower()
        s = _paren.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()

    s_sub, s_rel, s_obj = _norm(suggested.get("subject")), _norm(suggested.get("relation")), _norm(suggested.get("object"))
    o_sub, o_rel, o_obj = _norm(verdict_record.get("subject")), _norm(verdict_record.get("relation")), _norm(verdict_record.get("object"))
    flags = list(verdict_record.get("flags") or [])
    sub = verdict_record.get("revise_subtype")

    if s_sub and s_sub == s_obj:
        flags.append("sanitized_self_loop")
        verdict_record["suggested_triple"] = None
        verdict_record["revise_subtype"] = "4.10"
        verdict_record["flags"] = flags
        return

    if sub != "4.10" and (s_sub, s_rel, s_obj) == (o_sub, o_rel, o_obj):
        flags.append("sanitized_revise_no_change_reclassified")
        verdict_record["verdict"] = "accept_direct"
        verdict_record["suggested_triple"] = None
        verdict_record["revise_subtype"] = None
        verdict_record["stage_b_justification"] = None
        verdict_record["stage_b_confidence"] = None
        verdict_record["flags"] = flags
        return

    if s_sub not in (o_sub, o_obj) and s_obj not in (o_sub, o_obj):
        flags.append("sanitized_class_substitution")
        verdict_record["suggested_triple"] = None
        verdict_record["revise_subtype"] = "4.10"
        verdict_record["flags"] = flags
        return


# ── Triple extraction from ontology dict ────────────────────────────

def collect_triples(ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten an ontology dict into a list of judge-ready triples.

    Each output item has: kind ("relation"|"hierarchy"), subject, relation, object,
    evidence (string or empty). Hierarchy edges are emitted with relation="subClassOf".
    """
    items: List[Dict[str, Any]] = []

    for r in ontology.get("relations") or []:
        subj = (r.get("domain") or r.get("subject") or "").strip()
        rel = (r.get("label") or r.get("relation") or "").strip()
        obj = (r.get("range") or r.get("object") or "").strip()
        if not (subj and rel and obj):
            continue
        items.append({
            "kind": "relation",
            "subject": subj,
            "relation": rel,
            "object": obj,
            "evidence": (r.get("evidence") or "").strip(),
        })

    for h in ontology.get("hierarchy") or []:
        sub = (h.get("subClass") or h.get("subclass") or h.get("child") or "").strip()
        sup = (h.get("superClass") or h.get("superclass") or h.get("parent") or "").strip()
        if not (sub and sup):
            continue
        items.append({
            "kind": "hierarchy",
            "subject": sub,
            "relation": "subClassOf",
            "object": sup,
            "evidence": (h.get("evidence") or "").strip(),
        })

    return items


def build_class_lookup(ontology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """label → class dict (definition, evidence, synonyms)."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for c in ontology.get("classes") or []:
        lab = (c.get("label") or "").strip()
        if lab and lab not in lookup:
            lookup[lab] = c
    return lookup


# ── Main orchestration ──────────────────────────────────────────────

ProgressCallback = Callable[[int, int, str], None]


def _judge_delay_override() -> float:
    """Pacing override for judge LLM calls.

    Defaults to 0.5s — enough to avoid burst 429s from low-TPM tiers
    while still giving parallelism room to breathe. Override via
    JUDGE_REQUEST_DELAY_SECONDS in .env.
    """
    try:
        return max(0.0, float((os.getenv("JUDGE_REQUEST_DELAY_SECONDS") or "0.5").strip()))
    except ValueError:
        return 0.5


def _judge_max_workers() -> int:
    try:
        n = int((os.getenv("JUDGE_MAX_WORKERS") or "4").strip())
    except ValueError:
        n = 4
    return max(1, min(16, n))


def _judge_one_triple(
    index: int,
    triple: Dict[str, Any],
    class_lookup: Dict[str, Dict[str, Any]],
    client: LLMClient,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Judge a single triple (Stage A + optional Stage B).

    Returns (verdict_record, prompts_log_entries). Safe to run concurrently
    from worker threads — reads only from shared inputs, does not mutate.
    """
    prompts_log: List[Dict[str, Any]] = []

    # Stage A
    prompt_a = build_stage_a_prompt(triple, class_lookup)
    try:
        raw_a = client.generate(prompt_a, max_tokens=400)
    except Exception as e:  # noqa: BLE001 — judge must keep going on provider errors
        raw_a = ""
        stage_a = {
            "verdict": "reject",
            "justification": f"provider error: {e}",
            "confidence": 0.0,
            "flags": ["provider_error"],
        }
    else:
        stage_a = _coerce_stage_a(_parse_json_strict(raw_a))

    prompts_log.append({
        "triple_index": index,
        "stage": "A",
        "prompt": prompt_a,
        "raw_response": raw_a,
    })

    verdict_record: Dict[str, Any] = {
        "triple_index": index,
        "kind": triple["kind"],
        "subject": triple["subject"],
        "relation": triple["relation"],
        "object": triple["object"],
        "evidence": triple.get("evidence", ""),
        "verdict": stage_a["verdict"],
        "justification": stage_a["justification"],
        "confidence": stage_a["confidence"],
        "flags": stage_a["flags"],
        "revise_subtype": None,
        "suggested_triple": None,
        "stage_b_justification": None,
        "stage_b_confidence": None,
    }

    # Stage B (only for revise)
    if stage_a["verdict"] == "revise":
        prompt_b = build_stage_b_prompt(triple, class_lookup, stage_a["justification"])
        try:
            raw_b = client.generate(prompt_b, max_tokens=400)
        except Exception as e:  # noqa: BLE001
            raw_b = ""
            stage_b = {
                "revise_subtype": "4.10",
                "suggested_triple": None,
                "justification": f"provider error: {e}",
                "confidence": 0.0,
            }
        else:
            stage_b = _coerce_stage_b(_parse_json_strict(raw_b), triple)

        prompts_log.append({
            "triple_index": index,
            "stage": "B",
            "prompt": prompt_b,
            "raw_response": raw_b,
        })

        verdict_record["revise_subtype"] = stage_b["revise_subtype"]
        verdict_record["suggested_triple"] = stage_b["suggested_triple"]
        verdict_record["stage_b_justification"] = stage_b["justification"]
        verdict_record["stage_b_confidence"] = stage_b["confidence"]
        _sanitize_stage_b(verdict_record)

    return verdict_record, prompts_log


def run_qualitative_eval(
    ontology: Dict[str, Any],
    *,
    provider: str,
    model: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    max_triples: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Run the two-stage judge over every triple in an ontology.

    Triples are judged in parallel (default 4 workers) and the judge's
    LLM client bypasses the global extraction pacing (uses
    JUDGE_REQUEST_DELAY_SECONDS, default 0.5s) so it doesn't inherit
    long per-call delays from the OE pipeline's TPM throttle.

    Returns (verdicts, summary, prompts_log) with entries ordered by
    original triple index.
    """
    triples = collect_triples(ontology)
    if max_triples is not None and max_triples > 0:
        triples = triples[:max_triples]

    class_lookup = build_class_lookup(ontology)
    client = LLMClient(
        provider=provider,
        model=model,
        request_delay_override=_judge_delay_override(),
    )

    total = len(triples)
    if total == 0:
        return [], _empty_summary(provider, model), []

    workers = max(1, min(_judge_max_workers() if max_workers is None else int(max_workers), total))

    verdicts_by_index: Dict[int, Dict[str, Any]] = {}
    prompts_by_index: Dict[int, List[Dict[str, Any]]] = {}

    progress_lock = threading.Lock()
    completed_count = {"n": 0}

    def _on_done(idx: int, triple: Dict[str, Any]) -> None:
        with progress_lock:
            completed_count["n"] += 1
            n = completed_count["n"]
        if progress_callback:
            progress_callback(
                n, total,
                f"Judged {n}/{total}: {triple['subject']} — {triple['relation']} — {triple['object']}",
            )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qeval") as pool:
        future_to_idx = {
            pool.submit(_judge_one_triple, i, triple, class_lookup, client): (i, triple)
            for i, triple in enumerate(triples)
        }
        for future in as_completed(future_to_idx):
            i, triple = future_to_idx[future]
            try:
                verdict_record, log_entries = future.result()
            except Exception as e:  # noqa: BLE001 — never let one bad triple kill the run
                verdict_record = {
                    "triple_index": i,
                    "kind": triple["kind"],
                    "subject": triple["subject"],
                    "relation": triple["relation"],
                    "object": triple["object"],
                    "evidence": triple.get("evidence", ""),
                    "verdict": "reject",
                    "justification": f"worker error: {e}",
                    "confidence": 0.0,
                    "flags": ["worker_error"],
                    "revise_subtype": None,
                    "suggested_triple": None,
                    "stage_b_justification": None,
                    "stage_b_confidence": None,
                }
                log_entries = []
            verdicts_by_index[i] = verdict_record
            prompts_by_index[i] = log_entries
            _on_done(i, triple)

    # Reorder to original triple sequence for reproducibility.
    verdicts: List[Dict[str, Any]] = [verdicts_by_index[i] for i in range(total)]
    prompts_log: List[Dict[str, Any]] = []
    for i in range(total):
        prompts_log.extend(prompts_by_index.get(i, []))

    if progress_callback:
        progress_callback(total, total, "Judging complete.")

    summary = aggregate(verdicts, provider=provider, model=model)
    return verdicts, summary, prompts_log


# ── Aggregation ──────────────────────────────────────────────────────

# Action-weighted quality score weights. Higher = less engineer effort to accept.
# accept_direct is free; accept_indirect requires minor sanity-check;
# lexical-only revise is cheap; target/label revises are moderate;
# reject is zero.
_ACTION_WEIGHT = {
    "accept_direct": 1.0,
    "accept_indirect": 0.75,
    "reject": 0.0,
    # revise sub-types
    "4.6": 0.8,  # name-only fix — cheapest
    "4.8": 0.7,  # swap direction — cheap
    "4.1": 0.5, "4.2": 0.5, "4.3": 0.4,  # target moves
    "4.4": 0.5, "4.5": 0.45,             # relation edits
    "4.7": 0.4, "4.9": 0.4,              # kind conversions
    "4.10": 0.25,                        # unknown
}


def _empty_summary(provider: str, model: Optional[str]) -> Dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "n_triples": 0,
        "verdict_counts": {k: 0 for k in STAGE_A_LABELS},
        "verdict_pct": {k: 0.0 for k in STAGE_A_LABELS},
        "revise_breakdown_counts": {k: 0 for k in STAGE_B_LABELS},
        "revise_breakdown_pct": {k: 0.0 for k in STAGE_B_LABELS},
        "mean_confidence": 0.0,
        "low_confidence_count": 0,
        "action_weighted_quality": 0.0,
        "engineer_effort_count": 0,
    }


def aggregate(verdicts: List[Dict[str, Any]],
              *, provider: str, model: Optional[str] = None) -> Dict[str, Any]:
    summary = _empty_summary(provider, model)
    n = len(verdicts)
    if n == 0:
        return summary

    counts = {k: 0 for k in STAGE_A_LABELS}
    revise_counts = {k: 0 for k in STAGE_B_LABELS}
    conf_sum = 0.0
    low_conf = 0
    weighted = 0.0
    effort = 0

    for v in verdicts:
        verdict = v["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
        conf_sum += float(v.get("confidence") or 0.0)
        if float(v.get("confidence") or 0.0) < 0.5:
            low_conf += 1
        if verdict == "revise":
            sub = v.get("revise_subtype") or "4.10"
            revise_counts[sub] = revise_counts.get(sub, 0) + 1
            weighted += _ACTION_WEIGHT.get(sub, 0.25)
            effort += 1
        elif verdict == "reject":
            effort += 1
        else:
            weighted += _ACTION_WEIGHT.get(verdict, 0.0)

    summary.update({
        "n_triples": n,
        "verdict_counts": counts,
        "verdict_pct": {k: round(v * 100.0 / n, 2) for k, v in counts.items()},
        "revise_breakdown_counts": revise_counts,
        "revise_breakdown_pct": {k: round(v * 100.0 / n, 2) for k, v in revise_counts.items()},
        "mean_confidence": round(conf_sum / n, 3),
        "low_confidence_count": low_conf,
        "action_weighted_quality": round(weighted / n, 3),
        "engineer_effort_count": effort,
    })
    return summary


# ── CSV export ───────────────────────────────────────────────────────

_CSV_HEADER = [
    "triple_index", "kind", "subject", "relation", "object", "evidence",
    "verdict", "revise_subtype",
    "suggested_subject", "suggested_relation", "suggested_object",
    "justification", "confidence",
    "stage_b_justification", "stage_b_confidence",
    "flags",
]


def verdicts_to_csv(verdicts: List[Dict[str, Any]]) -> str:
    """Render verdicts as CSV. Uses simple quoting compatible with Excel."""
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    for v in verdicts:
        sug = v.get("suggested_triple") or {}
        writer.writerow([
            v.get("triple_index", ""),
            v.get("kind", ""),
            v.get("subject", ""),
            v.get("relation", ""),
            v.get("object", ""),
            v.get("evidence", ""),
            v.get("verdict", ""),
            v.get("revise_subtype") or "",
            sug.get("subject", "") if isinstance(sug, dict) else "",
            sug.get("relation", "") if isinstance(sug, dict) else "",
            sug.get("object", "") if isinstance(sug, dict) else "",
            v.get("justification", ""),
            v.get("confidence", ""),
            v.get("stage_b_justification") or "",
            v.get("stage_b_confidence") if v.get("stage_b_confidence") is not None else "",
            "|".join(v.get("flags") or []),
        ])
    return buf.getvalue()
