"""
Symbolic reasoning layer: deterministic post-processing to enforce schema constraints
and infer logically implied edges. Applied after merge (and optional schema-guided completion).

Pipeline position: Extraction → Merge → Vocabulary filter → [Schema-guided completion]
→ [LLM Reasoning Layer] → [Built-in cleanup] → [Rule-based Reasoning Layer (optional)] → Evaluation

Split into:
- BUILT-IN CLEANUP (always on): dedupe, domain/range prune, scope/evidence pruning.
  Cleanup is guided by the *domain scope file* (resources/domain_scope.json), which
  provides in-scope class/relation lists and allowlists curated from all 11 papers.
  This is independent of the gold-standard ontology (used for evaluation only).
  When gold_free=True (Strict mode), also skips gold-label canonicalization and axioms.
- RULE-BASED REASONING LAYER (optional via UI toggle): schema completion from gold,
  orphan pruning only (_complete_hierarchy_from_schema, _prune_orphan_classes).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .canonical import canonical_key, resolve_to_canonical_label
from .model import ClassEntity, Ontology, RelationEntity
from .neuro_axioms import (
    check_hierarchy_axiom,
    check_relation_axiom,
    infer_semantic_types_from_gold,
)

from ..prompting.vocabulary import CLASS_SYNONYM_MAP

_REVERSE_SYNONYM_MAP: Dict[str, List[str]] = {}
for _syn_key, _syn_val in CLASS_SYNONYM_MAP.items():
    if _syn_val is not None:
        _REVERSE_SYNONYM_MAP.setdefault(_syn_val.lower(), []).append(_syn_key)

# Corpus-derived abbreviations (loaded from corpus dictionary, replaces hardcoded list).
from ..corpus.corpus_vocab import (
    get_abbreviation_expansion as _corpus_get_abbr_expansion,
    get_synonym_group as _corpus_get_synonym_group,
    is_corpus_clinical_term as _corpus_is_clinical_term,
    get_term_paper_count as _corpus_get_term_paper_count,
    get_all_abbreviations as _corpus_get_all_abbreviations,
    corpus_dict_loaded as _corpus_dict_loaded,
)


# ── Lazy spaCy loader for semantic evidence matching ──────────────────────────
# Prefers en_core_sci_sm (biomedical; already installed) for better domain
# coverage. Falls back to any available model, or None if spaCy unavailable.
_SCI_NLP = None   # the loaded model, False if load failed


def _get_sci_nlp():
    """Lazy-load the biomedical spaCy model. Returns None if unavailable."""
    global _SCI_NLP
    if _SCI_NLP is not None:
        return None if _SCI_NLP is False else _SCI_NLP
    try:
        import spacy as _spacy
        for _m in ("en_core_sci_sm", "en_core_web_sm", "en_core_web_md"):
            try:
                _SCI_NLP = _spacy.load(_m, disable=["ner", "parser"])
                break
            except OSError:
                continue
        else:
            _SCI_NLP = False
    except ImportError:
        _SCI_NLP = False
    return None if _SCI_NLP is False else _SCI_NLP


def _lemma_set(text: str, nlp) -> set:
    """Return the set of lemmas for significant tokens in *text*."""
    doc = nlp(text.lower())
    return {
        tok.lemma_ for tok in doc
        if not tok.is_stop and not tok.is_punct and len(tok.lemma_) >= 3
    }


def _max_word_pair_similarity(label: str, ev_text: str, nlp) -> float:
    """Return the maximum cosine similarity across all (label_token, evidence_token) pairs.

    Using per-token similarity rather than doc-level similarity avoids dilution
    from non-matching tokens.  E.g. "Arterial Oxygen Saturation" vs "including
    pulse oximetry" gives 0.72 via the (saturation, oximetry) pair even though
    the doc-level similarity is only 0.30.

    Only non-stop tokens with vectors and length ≥ 3 are compared.
    Returns 0.0 if either side has no eligible tokens.
    """
    l_doc = nlp(label.lower())
    e_doc = nlp(ev_text.lower()[: _EVIDENCE_SIMILARITY_MAX_CHARS])
    l_toks = [t for t in l_doc if t.has_vector and not t.is_stop and len(t.text) >= 3]
    e_toks = [t for t in e_doc if t.has_vector and not t.is_stop and len(t.text) >= 3]
    if not l_toks or not e_toks:
        return 0.0
    return max(lt.similarity(et) for lt in l_toks for et in e_toks)


# Minimum best-word-pair cosine similarity to treat a label as evidenced.
# Uses max(sim(label_tok, ev_tok)) across all non-stop token pairs — this is
# much more robust than doc-level similarity because it finds the single
# most-semantically-similar word pair (e.g. "oximetry" ↔ "saturation" = 0.72).
# Threshold 0.65 validated against BrainIT papers:
#   - valid clinical concepts (Arterial Oxygen Saturation, Arterial Hypotension) score ≥ 0.67
#   - out-of-scope noise that passed earlier filters (Pilot Study, Summary Statistics) score < 0.65
_EVIDENCE_SIMILARITY_THRESHOLD = 0.65

# Maximum evidence length (chars) passed to spaCy — caps token count for speed.
_EVIDENCE_SIMILARITY_MAX_CHARS = 400


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _norm_label(s: str) -> str:
    return _norm(s)


# --- Out-of-scope class pruning (governance / admin / org / methodology / tech) ---
# Use compound phrases to avoid destroying legitimate clinical labels.
# Protected gold classes (never matched here): Data Quality Assessment, Possible Error,
# Probable Error, Observation, Sensor, Session, Timepoint, Parameter.
_OUT_OF_SCOPE_CLASS_PATTERNS = [
    # Governance / admin / organisational
    r"\bsteering group\b",
    r"\btechnical group\b",
    r"\bproject group",
    r"\bstaff\b",
    r"\bcommittee\b",
    r"\bmembership\b",
    r"\bcentre(s)?\b",
    r"\bdatabase\b",
    r"\bquality control\b",
    r"\bfeasibility\b",
    r"\bconfidentiality\b",
    r"\bownership\b",
    r"\bnon-profit\b",
    r"\bnot.for.profit\b",
    r"\borganisation\b",
    r"\borganization\b",
    # Semantic web / data technology (Moss 2013 noise)
    r"\blinked data\b",
    r"\brdf\b",
    r"\brdf triple",
    r"\bsemantic web\b",
    r"\bsparql\b",
    r"\bowl ontolog",
    r"\bontolog(y|ies)\b",
    r"\bprovenance (information|annotation)\b",
    r"\berror annotation\b",
    # Data methodology / validation (NOT "Data Quality Assessment" — gold!)
    r"\bdata validation\b",
    r"\bdata model\b",
    r"\bdata collection (software|methods?|tools?|form)\b",
    r"\bvalidated data\b",
    r"\banonymous data\b",
    r"\bdata analysis method",
    r"\bdata integration\b",
    # Research / study design
    r"\bresearch group",
    r"\bresearch\b$",
    r"\bprincipal investigator\b",
    r"\bhelsinki\b",
    r"\bpost.hoc hypothesis\b",
    r"\bpilot data collection\b",
    r"\bclinical drug trial",
    r"\bmultivariate analysis\b",
    r"\blogistic regression\b",
    r"\btransition curve",
    r"\bprognostic score",
    r"\bage band",
    # Technical / software / configuration
    r"\bsoftware tool",
    r"\bmedical device compan",
    r"\brecording configuration\b",
    r"\bconfiguration\b$",
    r"\berror rate",
    # Company / institution names
    r"\bphilips\b",
    r"\bcma microdialysis\b",
    # Rules / methodology labels (Moss 2013)
    r"\bmissing data rule\b",
    r"\bmedical disorders rule\b",
    r"\btreatments rule\b",
    r"\bcompleteness\b$",
    r"\bsyntactic.+accuracy\b",
    r"\bcommon data quality check",
    # Statistical methodology / analysis (papers 4, 7, 8, 9, 10, 11)
    r"\bsensitivity analysis\b",
    r"\bsample size\b",
    r"\btraining set\b",
    r"\btest set\b",
    r"\bcross.validation\b",
    r"\bleave.one.out\b",
    r"\bbootstrap\b",
    r"\broc curve\b",
    r"\breceiver operating\b",
    r"\barea under the curve\b",
    r"\bauc\b$",
    r"\bmonte carlo\b",
    r"\bkaplan.meier\b",
    r"\bmann.whitney\b",
    r"\bwilcoxon\b",
    r"\bfisher.s exact\b",
    r"\bshapiro.wilk\b",
    r"\bchi.square",
    r"\bbonferroni\b",
    r"\bone.way anova\b",
    r"\blogistic regression\b",
    r"\blinear regression\b",
    r"\bcubic regression\b",
    r"\bmultiple imputation\b",
    r"\bmarginal effects?\b",
    r"\bconfidence interval\b",
    r"\bdeviance test",
    r"\bp.value\b",
    r"\bfalse positive\b",
    r"\btrue positive\b",
    r"\bprediction model\b",
    r"\bpredictive model\b",
    r"\bprediction threshold\b",
    r"\bdecision threshold\b",
    r"\bover.?fitting\b",
    r"\bspecificity\b$",
    r"\bprecision\b$",
    r"\brecall\b$",
    r"\bsensitivity\b$",
    # Machine learning / neural network (paper 7, 9)
    r"\bneural network\b",
    r"\bartificial neural network\b",
    r"\bbayesian.+neural\b",
    r"\bbann\b",
    r"\bmachine learning\b",
    r"\bsupport vector\b",
    r"\bkalman filter\b",
    r"\bgaussian process\b",
    r"\bhypo.?net\b",
    r"\bhypo.?predict\b",
    # Software / tool names (papers 3-11)
    r"\bmatlab\b",
    r"\bspss\b",
    r"\bsmartpls\b",
    r"\bstatistica\b",
    r"\bicm\+",
    r"\bpowerlab\b",
    # Process mining / IT (paper 3)
    r"\bprocess model\b",
    r"\bprocess mining\b",
    r"\bbpmn\b",
    r"\buml\b$",
    r"\bevent log\b",
    r"\bdata.?push\b",
    r"\bward web app\b",
    r"\bdata warehouse\b",
    r"\bdata.?store\b",
    r"\binput definition syntax\b",
    # Phase 29: Additional out-of-scope patterns
    r"\bcohort\b$",
    r"\bprediction\b$",
    r"\bpredictive\b",
    r"\btemporal correlation\b",
    r"\bprocess models?\b$",
    r"\brandomised clinical trial\b",
    r"\bclinical trials?\b$",
    # Study design / ethics (papers 4, 9, 10)
    r"\bcohort study\b",
    r"\brandomized controlled\b",
    r"\brandomised controlled\b",
    r"\bprospective.+(study|observational)\b",
    r"\bretrospective.+(study|analysis)\b",
    r"\binclusion criteria\b",
    r"\bexclusion criteria\b",
    r"\binformed consent\b",
    r"\bethics.+approval\b",
    r"\bethics.+committee\b",
    r"\binstitutional review board\b",
    r"\bmeta.?analysis\b",
    r"\bclinical trial design\b",
    r"\bhanfelt design\b",
    r"\bblinded.endpoint\b",
    # Publishing / book metadata
    r"\bauthor index\b",
    r"\bsubject index\b",
    r"\bpreface\b$",
    r"\btable of contents\b",
    r"\bsupplementary (table|figure|material)",
    r"\belectronic supplementary\b",
    # Institution / publisher names (papers 3-11)
    r"\bspringer\b",
    r"\belsevier\b",
    r"\bscitepress\b",
    r"\bsciencedirect\b",
    r"\bacta neurochirurgica\b",
    r"\bstats research ltd\b",
    # Signal processing (papers 5, 7)
    r"\bfourier transform\b",
    r"\bwavelet analysis\b",
    r"\blinear interpolation\b",
    r"\bspectral analysis\b",
    r"\bsignal processing\b",
    r"\bcontour plot\b",
    r"\bcolour.coded plot\b",
    # Research project names
    r"\bcenter.tbi\b",
    r"\bavert.it\b",
    r"\bnemo project\b",
    r"\bdecra trial\b",
    r"\bbest.trip trial\b",
    r"\bimpact score\b",
    r"\bcrash prediction\b",
    r"\bcarnet\b",
    r"\biscope\b",
    # Data management / acquisition / infrastructure
    r"\bdata.?acquisition\b",
    r"\bdata capture\b",
    r"\bdata preparation\b",
    r"\bdata management\b",
    r"\bdata.?set\b$",
    r"\bdata element",
    r"\bdata contributing\b",
    r"\bdata quality control\b",
    r"\bdata analysis\b",
    r"\bdata analysis method",
    r"\bdata collection\b",
    r"\bdata validation\b",
    r"\bcore dataset\b",
    r"\btime.series data\b",
    # Governance / org / network / group concepts (Piper 2003, 2010)
    r"\bbrainit\b",
    r"\bproject management\b",
    r"\bproject group",
    r"\bpublication\b$",
    r"\bcollaborative group",
    r"\bregistration form\b",
    r"\binternet registration\b",
    r"\bresearch study\b",
    r"\bresearch studi",
    r"\bmulti.centre trial",
    r"\bmulticenter trial",
    r"\bmulticentre trial",
    r"\bnon.profit\b",
    r"\bnot.for.profit\b",
    r"\bnetwork\b$",
    r"\bfunding\b$",
    r"\bgrant fund",
    # Data / infrastructure concepts that survive as class labels
    r"\bdefinition file\b",
    r"\bsoftware\b$",
    r"\bcollection tool",
    r"\bcollection method",
    r"\bcollection protocol",
    r"\bcollection software\b",
    r"\bvalidation standard",
    r"\bvalidation study\b",
    r"\bvalidation staff\b",
    r"\bsampling technique",
    r"\bnursing chart\b$",
    r"\bbedside monitoring type",
    r"\bhealth care technolog",
    r"\bhealthcare technolog",
    # Patient attributes that should be data properties, not classes
    r"\bpatient age\b$",
    r"\bpatient sex\b$",
    r"\bpatient population\b$",
    r"\bpatient management\b$",
    # Vague/abstract concepts that don't belong in a clinical ontology
    r"\bclinical information\b$",
    r"\bphysiological monitoring\b$",
    r"\btherapy target",
    r"\bcpp management\b$",
    r"\bmotor score\b$",
    r"\bhypoxia evidence\b$",
    r"\bnot testable\b",
    r"\bunknown field\b",
    r"\bmissing data\b$",
    r"\bmissing error\b",
    r"\bassociated major injur",
    # Redundant description-style classes (section headings extracted as concepts)
    r"^demographic and clinical",
    r"^minute by minute",
    r"^intensive care management information",
    r"^secondary insult treatment information",
    r"^secondary insult management",
    # Generic infrastructure / IT
    r"\bsql database\b",
    r"\bweb.?site\b",
    r"\bweb.?page\b",
    r"\bforum\b$",
    r"\bmailbase\b",
    r"\binfrastructure\b$",
    # Institution / hospital / university / department names (across all 11 papers)
    r"\buniversity\b",
    r"\bhospital\b",
    r"\bdepartment\b",
    r"\bschool of\b",
    r"\bcollege of\b",
    r"\binstitute of\b",
    r"\bnhs\b",
    r"\bclinic\b$",
    r"\bmedical cent(re|er)\b",
    # Country / city names as classes
    r"\bglasgow\b$",
    r"\bedinburgh\b$",
    r"\buppsala\b$",
    r"\bleuven\b$",
    r"\bmonza\b$",
    r"\beuropean\b",
    # Project / consortium names (papers 3-11)
    r"\brescueicp\b",
    r"\bhypo.?net\b",
    r"\bnemo project\b",
    # Publishing / reference metadata classes
    r"\bjournal\b$",
    r"\bpublisher\b",
    r"\bcopyright\b",
    r"\beditor\b$",
    r"\bmanuscript\b",
    r"\bartifact type\b",
    r"\breference list\b",
    r"\bbibliograph",
    # Data result / collection result classes
    r"\bcollection result",
    r"\bdata result",
    r"\bstudy result",
    r"\bstudy finding",
    r"\bstudy outcome\b$",
    r"\bstudy population\b$",
    r"\bstudy group\b$",
    r"\bstudy cohort\b$",
    r"\bpatient cohort\b$",
    # Vague meta-concepts
    r"\binformation system\b",
    r"\bclinical variable\b$",
    r"\bphysiological variable\b$",
    r"\bmonitoring variable\b$",
    r"\bmonitoring system\b$",
    r"\bmonitoring device",
    r"\bmonitoring equipment",
    r"\brecording system\b",
    r"\bdata source\b$",
    r"\btime interval\b$",
    r"\bsignal quality\b$",
    r"\bneuroprotective drug",
]

# Evidence patterns that indicate governance/admin context (drop class even if label looks generic).
# Uses compound phrases to avoid destroying classes with evidence that merely mentions these words.
_OUT_OF_SCOPE_EVIDENCE_PATTERNS = [
    r"\bgovernance\b",
    r"\bsteering group\b",
    r"\bconsortium\b",
    r"\bcommittee\b",
    r"\bvalidation (workflow|process|criteria)\b",
    r"\bdata (collection validation|validation workflow)\b",
    r"\bethics (approval|committee)\b",
    r"\bconfidentiality\b",
    r"\bownership\b",
    r"\bpublication criteria\b",
    r"\bfunding\b",
    r"\bgroup (formation|coordination|membership)\b",
    r"\bmembership\b",
    r"\bcentre(s)?\b",
    r"\binfrastructure\b",
    r"\bwebsite\b",
    r"\bsoftware tool\b",
    r"\bdata collection form\b",
    r"\bpaper based pilot\b",
    # Semantic web / data technology
    r"\brdf\b",
    r"\blinked data\b",
    r"\bsemantic web\b",
    r"\bsparql\b",
    r"\bontology.based\b",
    r"\bprovenance annotation\b",
    # Research methodology
    r"\bprincipal investigator\b",
    r"\bhelsinki\b",
    r"\bclinical drug trial\b",
    r"\blogistic regression\b",
    r"\bmultivariate\b",
    r"\bstatistical.+analysis\b",
    # Statistical methodology evidence (papers 4, 7, 8, 9, 10, 11)
    r"\bcross.validation\b",
    r"\bleave.one.out\b",
    r"\bbootstrap\b",
    r"\bmonte carlo\b",
    r"\bchi.square\b",
    r"\bmann.whitney\b",
    r"\bwilcoxon\b",
    r"\bkaplan.meier\b",
    r"\bbonferroni\b",
    r"\bsensitivity analysis\b",
    r"\bsample size\b",
    r"\bprediction model\b",
    # Software evidence
    r"\bmatlab\b",
    r"\bspss\b",
    r"\bneural network\b",
    r"\bmachine learning\b",
    r"\bprocess mining\b",
    # Study design evidence
    r"\binclusion criteria\b",
    r"\bexclusion criteria\b",
    r"\binformed consent\b",
    r"\bethics committee\b",
    r"\binstitutional review\b",
    # Publishing evidence
    r"\bspringer\b",
    r"\belsevier\b",
    r"\bsciencedirect\b",
    # Network / group / project evidence
    r"\bbrainit group\b",
    r"\bbrainit network\b",
    r"\bbrainit project\b",
    r"\bproject management\b",
    r"\bregistration form\b",
    r"\bnon-profit\b",
    r"\bnot for profit\b",
    r"\bcollaboration\b",
    r"\bmailbase\b",
    r"\bforum\b",
    r"\bdata contributing\b",
    r"\bdata push\b",
]

# Abstract/data-framing labels (prune unless in allowlist).
# Use compound or specific patterns to avoid removing legitimate clinical classes.
_ABSTRACT_DATA_LABEL_PATTERNS = [
    r"\braw data\b",
    r"\bcore data\b",
    r"\bdataset\b",
    r"\belement(s)?\b",
    r"\btechnology\b",
    r"\bworkflow\b",
    r"\brule\b$",
    r"\bannotation\b$",
]

# Labels that may contain "data", "information", or "treatment" but are clinically valid.
# Loaded from domain scope file; falls back to hardcoded set if file unavailable.
def _get_abstract_label_allowlist() -> Set[str]:
    try:
        from .domain_scope import get_abstract_data_allowlist_norm
        ds = get_abstract_data_allowlist_norm()
        if ds:
            return set(ds)
    except Exception:
        pass
    return {
        _norm("Monitoring Data"), _norm("Demographic Data"),
        _norm("Core Monitoring Parameter"), _norm("Optional Monitoring Parameter"),
        _norm("Derived Parameter"), _norm("Data Quality Assessment"),
        _norm("Secondary Insult Treatment"), _norm("Laboratory Values"),
    }

# Broad contextual labels (not governance, but too generic for ontology). Prune unless in allowlist.
_BROAD_CONTEXT_LABEL_PATTERNS = [
    r"\bpatient care\b",
    r"\bhead injured patients?\b",
    r"\bbrain injured patients?\b",
    r"\bnew therapies?\b",
    r"\bmonitoring devices?\b",
    # "intensive care monitoring" removed: now canonicalized → "Monitoring Data" at merge time
]

# Classes that match broad contextual patterns but are legitimate clinical concepts.
# Loaded from domain scope file; falls back to hardcoded set if file unavailable.
def _get_broad_context_allowlist() -> Set[str]:
    try:
        from .domain_scope import get_broad_context_allowlist_norm
        ds = get_broad_context_allowlist_norm()
        if ds:
            return set(ds)
    except Exception:
        pass
    return {
        _norm("Patient"), _norm("Therapy"), _norm("Secondary Insult Treatment"),
        _norm("Intensive Care Management"), _norm("Secondary Insults"),
        _norm("Baseline Therapy"), _norm("Secondary Insult Therapy"),
        _norm("Nursing Intervention"), _norm("Routine Nursing Care"),
        _norm("Bedside Intervention"), _norm("Patient Transport"),
        _norm("Clinical Assessment"), _norm("GCS Assessment"),
        _norm("Pupil Assessment"), _norm("CT Scan Assessment"),
        _norm("Surgical Procedure"), _norm("Analgesia"), _norm("Paralysis"),
        _norm("Anti-hypertensives"), _norm("Anti-pyretics"),
        _norm("Hypothermia Therapy"), _norm("Osmotics"), _norm("Barbiturates"),
        _norm("Steroids"), _norm("Decompressive Craniectomy"),
        _norm("ICP Sensor Placement"), _norm("Evacuation of Mass Lesion"),
        _norm("Skull Fracture Elevation"), _norm("Extra Ventricular Drain Placement"),
        _norm("Removal of Foreign Body"), _norm("Anterior Fossa Repair"),
        _norm("Demographic Data"), _norm("Peripheral Temperature"),
        _norm("Cardiac Output"), _norm("NIBP"), _norm("Pressure Reactivity Index (PRx)"),
    }

# Global allowed relation labels (evidence-based pruning).
# Includes paper-wording labels, camelCase gold schema labels, and common LLM variants.
_ALLOWED_RELATION_LABELS_GLOBAL = {
    # Paper-wording (used in EXTRACTION_RELATION_LABELS and pool examples)
    "includes",
    "has_source",
    "treats",
    "has_target",
    "secondary_to",
    "is_a",
    # CamelCase gold schema labels – v1.0
    "receivestherapy",
    "targetscondition",
    "hasmonitoringdata",
    "indicatescondition",
    "monitoringindicatescondition",
    "hasdemographicdata",
    # v2.0 gold schema labels (camelCase and space-separated)
    "hasclinicalassessment",
    "has clinical assessment",
    "hasoutcome",
    "has outcome",
    "haslaboratoryvalue",
    "has laboratory value",
    "hasnursingintervention",
    "has nursing intervention",
    "hassession",
    "has session",
    "hastimepoint",
    "has timepoint",
    "hasobservation",
    "has observation",
    "measuresparameter",
    "measures parameter",
    "producedbysensor",
    "produced by sensor",
    "hasqualityassessment",
    "has quality assessment",
    "associatedwithcondition",
    "associated with condition",
    "affectedbytreatment",
    "affected by treatment",
    "monitoring indicates condition",
    "targets condition",
    "receives therapy",
    "has monitoring data",
    "hassurgicalprocedure",
    "has surgical procedure",
    # Common LLM-generated variants (expanded for better recall)
    "monitors",
    "measured by",
    "measured with",
    "indicates",
    "causes",
    "caused by",
    "associated with",
    "related to",
    "requires",
    "used for",
    "used in",
    "derived from",
    "part of",
    "consists of",
    "has parameter",
    "has value",
    "has component",
    "has demographic data",
    "has injury mechanism",
    "hasinjurymechanism",
    "has eusig grade",
    "haseusiggrade",
    "has mortality",
    "hasmortality",
    "has guideline adherence",
    "hasguidelineadherence",
    "triggers intervention",
    "triggersintervention",
    "triggers",
    "manages",
    "managed by",
    "administered to",
    "produces",
    "recorded by",
    "records",
    "detects",
    "predicts",
    "affects",
    "influences",
    "depends on",
    "correlated with",
    "assessed by",
    "assessed with",
    "classified as",
    "categorized as",
    "subclass of",
    "type of",
    "has type",
    "has category",
    "has sensor",
    "has treatment",
    "has therapy",
    "has intervention",
    "has assessment",
    "has data",
    "has condition",
    "has procedure",
    "occurs during",
    "occurs in",
    "precedes",
    "follows",
    "results in",
    "leads to",
    "prevented by",
    "contraindicated in",
    "measured at",
    "observed in",
    "observed during",
    "monitored by",
    "monitored with",
    "recorded during",
    "collected from",
    "collected during",
    "has demographic",
    "provides",
    "receives",
    "undergoes",
    "evaluates",
    "quantifies",
}

# Normalize common LLM relation label variants to canonical forms before whitelist check.
_RELATION_LABEL_NORMALIZATION: Dict[str, str] = {
    "monitors": "has monitoring data",
    "monitored by": "has monitoring data",
    "monitored with": "has monitoring data",
    "measured by": "measures parameter",
    "measured with": "measures parameter",
    "indicates": "monitoring indicates condition",
    "indicates condition": "monitoring indicates condition",
    "causes": "associated with condition",
    "caused by": "associated with condition",
    "associated with": "associated with condition",
    "related to": "associated with condition",
    "treats": "targets condition",
    "manages": "targets condition",
    "managed by": "receives therapy",
    "administered to": "receives therapy",
    "receives": "receives therapy",
    "undergoes": "receives therapy",
    "produces": "produced by sensor",
    "recorded by": "produced by sensor",
    "records": "has monitoring data",
    "detects": "monitoring indicates condition",
    "predicts": "monitoring indicates condition",
    "affects": "affected by treatment",
    "influences": "affected by treatment",
    "assessed by": "has clinical assessment",
    "assessed with": "has clinical assessment",
    "evaluates": "has clinical assessment",
    "triggers": "triggers intervention",
    "results in": "has outcome",
    "leads to": "has outcome",
    "resulting in": "has outcome",
    "has treatment": "receives therapy",
    "has therapy": "receives therapy",
    "has intervention": "has nursing intervention",
    "has assessment": "has clinical assessment",
    "has condition": "associated with condition",
    "has procedure": "has surgical procedure",
    "has sensor": "produced by sensor",
    "has data": "has monitoring data",
    "has demographic": "has demographic data",
    "quantifies": "measures parameter",
    "provides": "has monitoring data",
    "part of": "includes",
    "consists of": "includes",
    "has component": "includes",
    "type of": "is_a",
    "classified as": "is_a",
    "categorized as": "is_a",
    "subclass of": "is_a",
    "has type": "is_a",
    "has category": "is_a",
    # Phase 26: Ad-hoc relation aliases → canonical forms
    "affected by": "receives therapy",
    "affected by treatment": "receives therapy",
    "has outcome data": "has outcome",
    "associated with outcome": "has outcome",
    "has observation": "has monitoring data",
    "has reading": "has monitoring data",
    "at risk from": "associated with condition",
    "works in": "associated with condition",
    "has clear holddown": "has monitoring data",
    "has eusig event": "has monitoring data",
}

# Deterministic hierarchy correction: when the LLM maps a child directly to a
# grandparent/great-grandparent (flat mapping), remap to the correct immediate parent.
# Key = child (normalized), Value = correct immediate parent label.
_HIERARCHY_CORRECT_PARENT: Dict[str, str] = {
    # Monitoring parameters → Core Monitoring Parameter (NOT Monitoring Data)
    "Mean Arterial Pressure (MAP)": "Core Monitoring Parameter",
    "Intracranial Pressure (ICP)": "Core Monitoring Parameter",
    "Cerebral Perfusion Pressure (CPP)": "Core Monitoring Parameter",
    "Heart Rate": "Core Monitoring Parameter",
    "SaO2": "Core Monitoring Parameter",
    "Temperature": "Core Monitoring Parameter",
    "Respiration Rate": "Core Monitoring Parameter",
    # Optional Monitoring → Optional Monitoring Parameter (NOT Monitoring Data)
    "CVP": "Optional Monitoring Parameter",
    "EtCO2": "Optional Monitoring Parameter",
    "NIBP": "Optional Monitoring Parameter",
    "Peripheral Temperature": "Optional Monitoring Parameter",
    "PbrO2": "Optional Monitoring Parameter",
    "SjO2": "Optional Monitoring Parameter",
    "Cardiac Output": "Optional Monitoring Parameter",
    "Brain Temperature": "Optional Monitoring Parameter",
    "TCD": "Optional Monitoring Parameter",
    "Microdialysis": "Optional Monitoring Parameter",
    # Derived → Derived Parameter (NOT Monitoring Data)
    "Pressure Reactivity Index (PRx)": "Derived Parameter",
    # Baseline therapies → Baseline Therapy (NOT Therapy)
    "Sedation": "Baseline Therapy",
    "Analgesia": "Baseline Therapy",
    "Paralysis": "Baseline Therapy",
    "Fluids": "Baseline Therapy",
    "Vasopressors": "Baseline Therapy",
    "Anti-hypertensives": "Baseline Therapy",
    "Anti-pyretics": "Baseline Therapy",
    "Hypothermia Therapy": "Baseline Therapy",
    "Ventilation": "Baseline Therapy",
    "Nutrition": "Baseline Therapy",
    "Antibiotics": "Baseline Therapy",
    # Secondary insult therapies → Secondary Insult Therapy (NOT Therapy)
    "Osmotics": "Secondary Insult Therapy",
    "Barbiturates": "Secondary Insult Therapy",
    "Steroids": "Secondary Insult Therapy",
    # Surgical subtypes → Surgical Procedure
    "ICP Sensor Placement": "Surgical Procedure",
    "Evacuation of Mass Lesion": "Surgical Procedure",
    "Skull Fracture Elevation": "Surgical Procedure",
    "Extra Ventricular Drain Placement": "Surgical Procedure",
    "Decompressive Craniectomy": "Surgical Procedure",
    "Removal of Foreign Body": "Surgical Procedure",
    "Anterior Fossa Repair": "Surgical Procedure",
    # Clinical assessment subtypes → Clinical Assessment
    "GCS Assessment": "Clinical Assessment",
    "Pupil Assessment": "Clinical Assessment",
    "CT Scan Assessment": "Clinical Assessment",
    # Conditions → Secondary Insult (NOT Condition)
    "Arterial Hypotension": "Secondary Insult",
    "Intracranial Hypertension": "Secondary Insult",
    "Systemic Hypotension": "Secondary Insult",
    "Jugular Venous Desaturation": "Secondary Insult",
    "Acute Respiratory Distress Syndrome": "Secondary Insult",
    # Lab subtypes → specific parent (NOT Laboratory Values directly for some)
    "Haemoglobin": "Haematology",
    "White Blood Cell Count": "Haematology",
    "Haematocrit": "Haematology",
    "Sodium": "Biochemistry",
    "Potassium": "Biochemistry",
    "Glucose": "Biochemistry",
    # Nursing subtypes → Nursing Intervention
    "Routine Nursing Care": "Nursing Intervention",
    "Physiotherapy": "Nursing Intervention",
    "Bedside Intervention": "Nursing Intervention",
    "Patient Transport": "Nursing Intervention",
    # Outcome
    "GOSe Outcome": "Outcome",
    # Data quality
    "Possible Error": "Data Quality Assessment",
    "Probable Error": "Data Quality Assessment",
}

# Build normalized lookup for fast matching
_HIERARCHY_CORRECT_PARENT_NORM: Dict[str, tuple[str, str]] = {}
for _child_lbl, _parent_lbl in _HIERARCHY_CORRECT_PARENT.items():
    _HIERARCHY_CORRECT_PARENT_NORM[_norm(_child_lbl)] = (_child_lbl, _parent_lbl)


# Hierarchy: reject if subclass/superclass contain these tokens (verb phrases, clauses)
_HIERARCHY_BAD_TOKENS = {
    "is", "are", "was", "were", "have", "has", "been",
    "occur", "occurs", "found", "more", "than", "realised",
    "because", "which", "that",
}

# Hierarchy: reject if starts with these (stopwords / non-NP)
_HIERARCHY_START_STOPWORDS = {"and", "or", "in", "of", "the", "a", "an"}

# Max tokens for a clean hierarchy node (noun phrase)
_HIERARCHY_MAX_TOKENS = 8

# Hierarchy: reject if node contains these phrases (narrative/example, not real is-a)
_HIERARCHY_BAD_PHRASES = [
    "good example",
    "for example",
    "e.g.",
    "such as",
    "e.g",
]

# Min token overlap ratio (label tokens in evidence) to keep class when literal match fails.
# Slightly relaxed (0.4) to recover recall while still dropping unsupported classes.
_CLASS_EVIDENCE_TOKEN_OVERLAP_THRESHOLD = 0.4

# Min evidence length for class to be kept (relaxed from 12 to 8 to recover recall).
_CLASS_EVIDENCE_MIN_LENGTH = 8

# Structural evidence exemptions: generic clinical ontology anchors protected from
# evidence pruning. In gold_free mode, this set is dynamically expanded by corpus
# frequency (terms in 5+ papers are considered structural clinical vocabulary).
# The base set contains universal abstractions that are always kept.
_STRUCTURAL_EVIDENCE_EXEMPTIONS_BASE: frozenset = frozenset({
    # Universal ontology anchors — always implicit in clinical text
    _norm("Condition"),
    _norm("Patient"),
    _norm("Therapy"),
    _norm("Treatment"),
    _norm("Intervention"),
    _norm("Procedure"),
    _norm("Assessment"),
    _norm("Observation"),
    _norm("Outcome"),
    _norm("Clinical Event"),
    _norm("Event"),
    _norm("Episode"),
    # Monitoring / measurement concepts — referenced indirectly
    _norm("Monitoring Data"),
    _norm("Parameter"),
    _norm("Physiological Parameter"),
    _norm("Physiological Data"),
    _norm("Physiological Signal"),
    _norm("Clinical Measurement"),
    _norm("Measurement"),
    _norm("Signal"),
    _norm("Waveform"),
    _norm("Sensor"),
    _norm("Monitor"),
    # Temporal / structural anchors
    _norm("Session"),
    _norm("Timepoint"),
    _norm("Time Series"),
    # Gold schema anchors (BrainIT-specific but universal in neuro-ICU papers)
    _norm("Clinical Assessment"),
    _norm("Nursing Intervention"),
    _norm("Nursing Procedure"),
    _norm("Laboratory Values"),
    _norm("Surgical Procedure"),
    _norm("Secondary Insult"),
    _norm("Intensive Care Management"),
    _norm("Demographic Data"),
    _norm("Data Quality Assessment"),
    # Generic relation endpoints that are always implicit
    _norm("Clinical Data"),
    _norm("Medical Record"),
    _norm("Handling Episode"),
})

_CORPUS_EVIDENCE_EXEMPTIONS_CACHE: frozenset | None = None


def _get_structural_evidence_exemptions() -> frozenset:
    """Return the structural evidence exemptions set, dynamically expanded with
    high-frequency corpus clinical terms (appearing in 5+ papers).

    The expanded set is cached after first computation.
    """
    global _CORPUS_EVIDENCE_EXEMPTIONS_CACHE
    if _CORPUS_EVIDENCE_EXEMPTIONS_CACHE is not None:
        return _CORPUS_EVIDENCE_EXEMPTIONS_CACHE

    expanded = set(_STRUCTURAL_EVIDENCE_EXEMPTIONS_BASE)
    if _corpus_dict_loaded():
        from ..corpus.corpus_vocab import get_all_clinical_terms
        for term, info in get_all_clinical_terms().items():
            if info.get("count", 0) >= 5:
                expanded.add(_norm(term))
    _CORPUS_EVIDENCE_EXEMPTIONS_CACHE = frozenset(expanded)
    return _CORPUS_EVIDENCE_EXEMPTIONS_CACHE

# Evidence pruning exemptions: loaded from domain scope file.
# Classes whose canonical labels rarely appear literally in paper text.
# Falls back to structural exemptions if domain scope unavailable.
def _get_evidence_pruning_exemptions() -> frozenset:
    try:
        from .domain_scope import get_evidence_exemptions_norm
        ds = get_evidence_exemptions_norm()
        if ds:
            return ds
    except Exception:
        pass
    return _STRUCTURAL_EVIDENCE_EXEMPTIONS_BASE


def _tokens(s: str) -> List[str]:
    """Return list of normalised alphanumeric tokens from s."""
    return [x for x in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if len(x) > 1 or x.isdigit()]


def _token_overlap_ratio(label_tokens: List[str], evidence_tokens: List[str]) -> float:
    """Fraction of label tokens that appear in evidence (0..1)."""
    if not label_tokens:
        return 1.0
    ev_set = set(evidence_tokens)
    return sum(1 for t in label_tokens if t in ev_set) / len(label_tokens)


def apply_post_pass_scope_cleanup(ontology) -> Dict:
    """Lightweight scope cleanup for use after TGC / orphan rescue passes.

    Removes out-of-scope classes and prunes relations/hierarchy edges that
    reference them.  Returns counts of removed items.
    """
    removed_classes = _prune_out_of_scope_classes(ontology)
    removed_classes += _prune_abstract_data_labels(ontology)
    removed_classes += _prune_broad_contextual_classes(ontology)
    _prune_edges_with_out_of_scope_endpoints(ontology)
    return {"out_of_scope_removed": removed_classes}


def _convert_includes_to_hierarchy(
    ontology: Ontology,
    *,
    audit_log: Optional[List] = None,
) -> int:
    """Convert 'includes' and 'is a type of' relations to hierarchy edges.

    When the LLM outputs ``includes(Domain, Range)`` it almost always means
    ``Range subClassOf Domain``.  Similarly ``is a type of(A, B)`` means
    ``A subClassOf B``.  We detect these, add the hierarchy edge, and remove
    the relation.  Returns the number of converted relations.

    Phase 24 of the Ontology Quality Improvement Plan.
    """
    _INCLUDES_PATTERNS = re.compile(
        r"^(includes?|consists?\s*of|comprises?|encompasses?"
        r"|is\s+composed\s+of|composed\s+of|is\s+made\s+up\s+of|made\s+up\s+of"
        r"|contains"
        r"|is\s+a\s+(type|kind|form|subtype|subclass|category)\s+of"
        r"|categorized\s+(as|under)|classified\s+(as|under)"
        r"|falls?\s+under|belongs?\s+to|part\s+of)$",
        re.IGNORECASE,
    )

    _REVERSED_DIRECTION = re.compile(
        r"^(is\s+a\s+(type|kind|form|subtype|subclass|category)\s+of"
        r"|categorized\s+(as|under)|classified\s+(as|under)"
        r"|falls?\s+under|belongs?\s+to|part\s+of)$",
        re.IGNORECASE,
    )

    surviving_class_labels = {
        canonical_key(resolve_to_canonical_label(c.label or "")[0])
        for c in ontology.classes
    }

    existing_hierarchy_keys = set()
    for e in ontology.hierarchy:
        sub = canonical_key(resolve_to_canonical_label((e.get("subClass") or "").strip())[0])
        sup = canonical_key(resolve_to_canonical_label((e.get("superClass") or "").strip())[0])
        if sub and sup:
            existing_hierarchy_keys.add((sub, sup))

    kept_relations: List[RelationEntity] = []
    converted = 0

    for r in ontology.relations:
        lab = (r.label or "").strip()
        if not _INCLUDES_PATTERNS.match(lab):
            kept_relations.append(r)
            continue

        domain_raw = (r.domain or "").strip()
        range_raw = (r.range or "").strip()
        if not domain_raw or not range_raw:
            kept_relations.append(r)
            continue

        domain_resolved = resolve_to_canonical_label(domain_raw)[0]
        range_resolved = resolve_to_canonical_label(range_raw)[0]
        domain_key = canonical_key(domain_resolved)
        range_key = canonical_key(range_resolved)

        if not domain_key or not range_key:
            kept_relations.append(r)
            continue

        # Both endpoints must be in the class list
        if domain_key not in surviving_class_labels or range_key not in surviving_class_labels:
            kept_relations.append(r)
            continue

        # Determine hierarchy direction:
        #   "includes(Domain, Range)" → Range subClassOf Domain
        #   "is a type of(A, B)"     → A subClassOf B
        #   "categorized as(A, B)"   → A subClassOf B
        #   "part of(A, B)"          → A subClassOf B
        is_reversed = bool(_REVERSED_DIRECTION.match(lab))
        if is_reversed:
            sub_label, sup_label = domain_resolved, range_resolved
            sub_key, sup_key = domain_key, range_key
        else:
            sub_label, sup_label = range_resolved, domain_resolved
            sub_key, sup_key = range_key, domain_key

        # Skip self-loops and already-existing edges
        if sub_key == sup_key:
            kept_relations.append(r)
            continue
        if (sub_key, sup_key) in existing_hierarchy_keys:
            # Edge already exists — just drop the relation
            converted += 1
            if audit_log is not None:
                audit_log.append({
                    "type": "relation", "label": lab,
                    "domain": domain_raw, "range": range_raw,
                    "stage": "includes_to_hierarchy",
                    "reason": f"Relation '{lab}' converted to existing hierarchy edge {sub_label} subClassOf {sup_label}",
                })
            continue

        # Add new hierarchy edge
        ontology.hierarchy.append({
            "subClass": sub_label,
            "superClass": sup_label,
            "evidence": getattr(r, "evidence", None) or f"Inferred from relation: {lab}({domain_raw}, {range_raw})",
            "provenance": list(r.provenance) + ["includes_to_hierarchy"],
            "stratum": getattr(r, "stratum", None) or "inferred",
        })
        existing_hierarchy_keys.add((sub_key, sup_key))
        converted += 1

        if audit_log is not None:
            audit_log.append({
                "type": "relation", "label": lab,
                "domain": domain_raw, "range": range_raw,
                "stage": "includes_to_hierarchy",
                "reason": f"Converted to hierarchy: {sub_label} subClassOf {sup_label}",
            })

    ontology.relations.clear()
    ontology.relations.extend(kept_relations)
    return converted


def apply_builtin_cleanup(
    ontology: Ontology,
    gold_schema: Dict,
    *,
    enable_abstract_data_pruning: bool = True,
    cleanup_config: Optional[Dict] = None,
    gold_free: bool = False,
) -> Dict:
    """
    Built-in post-processing cleanup.

    Cleanup is guided by the **domain scope file** (resources/domain_scope.json),
    which provides in-scope classes, evidence exemptions, and allowlists curated
    from all 11 BrainIT papers. This is independent of the gold-standard ontology,
    which is used for evaluation metrics only.

    Individual cleanup groups can be toggled off via *cleanup_config* dict.
    Recognised keys (all default True):
      cleanup_dedupe, cleanup_scope_pruning, cleanup_evidence_pruning,
      cleanup_structural, cleanup_axioms.

    Order: dedupe → out-of-scope classes → abstract data labels → broad contextual labels
    → class evidence → relation domain/range (against surviving classes) → relation evidence
    → hierarchy fragments → axiom constraints.

    gold_free: when True (Strict pipeline mode), also skips gold-label canonicalization
    and axiom constraints. Domain scope is still used for allowlists and exemptions.
    """
    # Domain scope provides cleanup guidance independently of gold standard.
    # Gold schema is only needed for label canonicalization (when gold_free=False).

    cfg = cleanup_config or {}
    do_dedupe = cfg.get("cleanup_dedupe", True)
    do_scope = cfg.get("cleanup_scope_pruning", True)
    do_evidence = cfg.get("cleanup_evidence_pruning", True)
    do_structural = cfg.get("cleanup_structural", True)
    do_axioms = cfg.get("cleanup_axioms", True) and not gold_free

    audit_log: List[Dict] = []

    gold_schema = gold_schema or {}
    gold_label_by_norm: Dict[str, str] = {}
    if not gold_free:
        gold_classes = gold_schema.get("classes", [])
        for c in gold_classes:
            lab = (c.get("label") or "").strip()
            if not lab:
                continue
            resolved = resolve_to_canonical_label(lab)[0]
            key = canonical_key(resolved)
            if key:
                gold_label_by_norm[key] = lab

    result: Dict = {}

    # ── Phase 24: Convert "includes" / "is a type of" relations to hierarchy edges ──
    result["includes_to_hierarchy"] = _convert_includes_to_hierarchy(ontology, audit_log=audit_log)

    # ── Relation label normalization (before range normalization and dedup) ──
    rel_labels_normalized = 0
    for r in ontology.relations:
        lab = (r.label or "").strip()
        normalized = _normalize_relation_label(lab)
        if normalized != lab:
            r.label = normalized
            rel_labels_normalized += 1
    result["relation_labels_normalized"] = rel_labels_normalized

    # ── Relation range normalization (before dedup so collapsed ranges get deduped) ──
    result["relations_range_normalized"] = _normalize_relation_ranges(ontology)

    # ── Deduplication ──
    if do_dedupe:
        n0 = len(ontology.classes)
        _dedupe_classes(ontology, gold_label_by_norm, gold_free=gold_free)
        result["classes_removed_dedupe"] = n0 - len(ontology.classes)

        n1 = len(ontology.relations)
        _dedupe_relations(ontology, gold_label_by_norm)
        result["relations_removed_dedupe"] = n1 - len(ontology.relations)

        n2 = len(ontology.hierarchy)
        _dedupe_hierarchy(ontology, gold_label_by_norm)
        result["hierarchy_removed_dedupe"] = n2 - len(ontology.hierarchy)
    else:
        result["classes_removed_dedupe"] = 0
        result["relations_removed_dedupe"] = 0
        result["hierarchy_removed_dedupe"] = 0

    # Snapshot class keys before pruning so we can block auto-endpoint
    # recovery from re-adding classes that cleanup explicitly removed.
    _pre_prune_class_keys: Set[str] = {
        canonical_key(resolve_to_canonical_label(c.label or "")[0]) for c in ontology.classes
    }

    # ── Scope / abstract pruning ──
    if do_scope:
        result["out_of_scope_classes_removed"] = _prune_out_of_scope_classes(ontology, audit_log=audit_log)
        if enable_abstract_data_pruning:
            result["abstract_data_classes_removed"] = _prune_abstract_data_labels(ontology, audit_log=audit_log)
        else:
            result["abstract_data_classes_removed"] = 0
        result["broad_contextual_classes_removed"] = _prune_broad_contextual_classes(ontology, audit_log=audit_log)
        _prune_edges_with_out_of_scope_endpoints(ontology, audit_log=audit_log)
    else:
        result["out_of_scope_classes_removed"] = 0
        result["abstract_data_classes_removed"] = 0
        result["broad_contextual_classes_removed"] = 0

    # ── Evidence pruning ──
    if do_evidence:
        result["classes_removed_evidence"] = _prune_classes_by_evidence(ontology, gold_free=gold_free, audit_log=audit_log)
    else:
        result["classes_removed_evidence"] = 0

    surviving_class_norm: Set[str] = {
        canonical_key(resolve_to_canonical_label(c.label or "")[0]) for c in ontology.classes
    }

    # Keys that were explicitly removed by scope/evidence pruning —
    # auto-endpoint recovery must NOT re-add these.
    _removed_class_keys = _pre_prune_class_keys - surviving_class_norm

    # ── Structural validation ──
    if do_structural:
        if not gold_free:
            endpoints_added = _auto_add_missing_endpoints(
                ontology, surviving_class_norm, removed_class_keys=_removed_class_keys,
            )
        else:
            endpoints_added = _auto_add_missing_endpoints_gold_free(
                ontology, surviving_class_norm, removed_class_keys=_removed_class_keys,
            )
        result["endpoints_auto_added"] = endpoints_added
        if endpoints_added > 0:
            result["out_of_scope_classes_removed"] += _prune_out_of_scope_classes(ontology, audit_log=audit_log)
            if enable_abstract_data_pruning:
                result["abstract_data_classes_removed"] += _prune_abstract_data_labels(ontology, audit_log=audit_log)
            result["broad_contextual_classes_removed"] += _prune_broad_contextual_classes(ontology, audit_log=audit_log)
            surviving_class_norm = {
                canonical_key(resolve_to_canonical_label(c.label or "")[0]) for c in ontology.classes
            }

        n_rels_scope = len(ontology.relations)
        n_hier_scope = len(ontology.hierarchy)
        _prune_edges_with_out_of_scope_endpoints(ontology, audit_log=audit_log)
        result["relations_pruned_scope"] = n_rels_scope - len(ontology.relations)
        result["hierarchy_pruned_scope"] = n_hier_scope - len(ontology.hierarchy)

        surviving_class_norm = {
            canonical_key(resolve_to_canonical_label(c.label or "")[0]) for c in ontology.classes
        }

        n_h_before = len(ontology.hierarchy)
        _prune_hierarchy_dangling_endpoints(ontology, surviving_class_norm, audit_log=audit_log)
        result["hierarchy_pruned_dangling"] = n_h_before - len(ontology.hierarchy)

        n3 = len(ontology.relations)
        _prune_relations_domain_range(ontology, surviving_class_norm, audit_log=audit_log)
        result["relations_pruned"] = n3 - len(ontology.relations)

        _gold_relation_labels_norm: Set[str] = set()
        if not gold_free:
            _gold_relation_labels_norm = {
                _norm(r.get("label", "")) for r in gold_schema.get("relations", []) if r.get("label")
            }
        result["relations_removed_evidence"] = _prune_weak_relations_by_evidence(
            ontology, gold_relation_labels_norm=_gold_relation_labels_norm, audit_log=audit_log,
        )
        result["hierarchy_fragments_removed"] = _prune_bad_hierarchy_fragments(ontology, audit_log=audit_log)
        result["hierarchy_edges_corrected"] = _correct_flat_hierarchy(ontology)

        if result["hierarchy_edges_corrected"] > 0:
            n_post_correct = len(ontology.hierarchy)
            _dedupe_hierarchy(ontology, gold_label_by_norm)
            post_correct_dedup = n_post_correct - len(ontology.hierarchy)
            result["hierarchy_removed_dedupe"] += post_correct_dedup

        # ── Phase 27.4: Domain-scope hierarchy scaffolding ──
        result["hierarchy_scaffolded"] = _scaffold_hierarchy_from_domain_scope(ontology, audit_log=audit_log)
        if result["hierarchy_scaffolded"] > 0:
            _dedupe_hierarchy(ontology, gold_label_by_norm)
    else:
        result["endpoints_auto_added"] = 0
        result["hierarchy_pruned_dangling"] = 0
        result["relations_pruned"] = 0
        result["relations_pruned_scope"] = 0
        result["hierarchy_pruned_scope"] = 0
        result["relations_removed_evidence"] = 0
        result["hierarchy_fragments_removed"] = 0
        result["hierarchy_edges_corrected"] = 0
        result["hierarchy_scaffolded"] = 0

    # ── Axiom constraints ──
    if do_axioms:
        violations: List[Dict] = []
        n6_h = len(ontology.hierarchy)
        n6_r = len(ontology.relations)
        _apply_axiom_constraints(ontology, gold_schema, violations)
        result["hierarchy_removed_axiom"] = n6_h - len(ontology.hierarchy)
        result["relations_removed_axiom"] = n6_r - len(ontology.relations)
        if violations:
            result["axiom_violations"] = violations
            for v in violations:
                if v.get("kind") == "hierarchy":
                    audit_log.append({"type": "hierarchy", "subClass": v.get("subClass"), "superClass": v.get("superClass"), "stage": "axiom_constraint", "reason": v.get("message", "axiom violation")})
                elif v.get("kind") == "relation":
                    audit_log.append({"type": "relation", "label": v.get("relation"), "domain": v.get("domain"), "range": v.get("range"), "stage": "axiom_constraint", "reason": v.get("message", "axiom violation")})
    else:
        result["hierarchy_removed_axiom"] = 0
        result["relations_removed_axiom"] = 0

    # ── Circular hierarchy detection ──
    result["hierarchy_cycles_removed"] = _remove_hierarchy_cycles(ontology, audit_log=audit_log)

    # ── Phase 27.5: Hierarchy completeness check ──
    result["hierarchy_completeness"] = _log_hierarchy_completeness(ontology)

    if audit_log:
        result["cleanup_audit_log"] = audit_log

    return result


def apply_symbolic_reasoner_enrichment(ontology: Ontology, gold_schema: Dict) -> Dict:
    """
    Optional Rule-based Reasoning Layer (UI toggle). Only schema completion and orphan pruning.

    Runs: _complete_hierarchy_from_schema, _prune_orphan_classes.
    Call after apply_builtin_cleanup when the user enables the "Rule-based Reasoning" toggle.

    Gold-aligned classes are protected from orphan pruning: only noise classes (no gold
    alignment and no structural connections) are removed.
    """
    if not gold_schema:
        return {"hierarchy_added": 0, "orphans_removed": 0}

    gold_classes = gold_schema.get("classes", [])
    gold_hierarchy = gold_schema.get("hierarchy", [])

    # Build gold class norm set for orphan-pruning protection
    gold_class_norms: Set[str] = set()
    for gc in gold_classes:
        lbl = (gc.get("label") or "").strip()
        if lbl:
            gold_class_norms.add(canonical_key(resolve_to_canonical_label(lbl)[0]))

    gold_hierarchy_set: Set[tuple] = set()
    for e in gold_hierarchy:
        sub, sup = (e.get("subClass") or "").strip(), (e.get("superClass") or "").strip()
        if not sub or not sup:
            continue
        sub_res = resolve_to_canonical_label(sub)[0]
        sup_res = resolve_to_canonical_label(sup)[0]
        k1, k2 = canonical_key(sub_res), canonical_key(sup_res)
        if k1 and k2:
            gold_hierarchy_set.add((k1, k2))

    n4 = len(ontology.hierarchy)
    _complete_hierarchy_from_schema(ontology, gold_hierarchy_set)
    hierarchy_added = len(ontology.hierarchy) - n4

    n5 = len(ontology.classes)
    _prune_orphan_classes(ontology, gold_class_norms=gold_class_norms)
    orphans_removed = n5 - len(ontology.classes)

    return {"hierarchy_added": hierarchy_added, "orphans_removed": orphans_removed}


def _dedupe_classes(ontology: Ontology, gold_label_by_norm: Dict[str, str],
                    *, gold_free: bool = False) -> None:
    """Dedupe classes by canonical key; merge provenance and aliases when merging duplicates.

    When *gold_free* is True and *gold_label_by_norm* is empty, corpus synonym
    groups are used to identify and merge duplicate concepts (e.g., merging
    "ICP" and "Intracranial Pressure" into a single class).
    """
    # In gold_free mode, build a corpus-based synonym → canonical key mapping
    corpus_syn_key: Dict[str, str] = {}
    corpus_syn_canonical: Dict[str, str] = {}
    if gold_free and _corpus_dict_loaded():
        from ..corpus.corpus_vocab import get_all_synonym_groups
        for group in get_all_synonym_groups():
            canon = group.get("canonical", "")
            if not canon:
                continue
            canon_key = canonical_key(canon)
            if not canon_key:
                continue
            for variant in group.get("variants", []):
                vk = canonical_key(variant)
                if vk:
                    corpus_syn_key[vk] = canon_key
                    corpus_syn_canonical[canon_key] = canon

    seen_norm: Dict[str, int] = {}
    kept: List[ClassEntity] = []
    for c in ontology.classes:
        resolved = resolve_to_canonical_label(c.label or "")[0]
        n = canonical_key(resolved)
        if not n:
            continue
        # In gold_free mode, map synonym variants to their canonical key
        if gold_free and n in corpus_syn_key:
            n = corpus_syn_key[n]
        if n in seen_norm:
            idx = seen_norm[n]
            existing = kept[idx]
            existing.provenance = list(dict.fromkeys(existing.provenance + list(c.provenance)))
            existing.aliases = list(dict.fromkeys(
                list(getattr(existing, "aliases", None) or [])
                + list(getattr(c, "aliases", None) or [])
                + ([c.label] if c.label and c.label != existing.label and c.label not in (existing.aliases or []) else [])
            ))
            if c.definition and not existing.definition:
                existing.definition = c.definition
            if getattr(c, "evidence", None) and (not getattr(existing, "evidence", None) or len((c.evidence or "")) > len((existing.evidence or ""))):
                existing.evidence = c.evidence
            continue
        seen_norm[n] = len(kept)
        # Pick canonical label: gold > corpus synonym > extracted
        if gold_label_by_norm:
            canonical = gold_label_by_norm.get(n, c.label or resolved)
        elif gold_free and n in corpus_syn_canonical:
            canonical = corpus_syn_canonical[n]
        else:
            canonical = c.label or resolved
        aliases = list(getattr(c, "aliases", None) or [])
        if c.label and c.label != canonical and c.label not in aliases:
            aliases.append(c.label)
        kept.append(
            ClassEntity(
                label=canonical,
                definition=c.definition,
                synonyms=list(c.synonyms) if c.synonyms else [],
                provenance=list(c.provenance) + ["symbolic_reasoner_dedupe"],
                original_label=c.original_label,
                evidence=getattr(c, "evidence", None),
                aliases=aliases,
                stratum=getattr(c, "stratum", None),
            )
        )
    ontology.classes.clear()
    ontology.classes.extend(kept)


_RELATION_RANGE_SUPERCLASS: Dict[str, str] = {
    "has monitoring data": "Monitoring Data",
    "receives therapy": "Therapy",
    "has surgical procedure": "Surgical Procedure",
    "has clinical assessment": "Clinical Assessment",
    "has outcome": "Outcome",
    "has laboratory value": "Laboratory Values",
    "has nursing intervention": "Nursing Intervention",
    "monitoring indicates condition": "Condition",
    "targets condition": "Condition",
    "associated with condition": "Condition",
    "triggers intervention": "Therapy",
    "affected by treatment": "Therapy",
    "has demographic data": "Demographic Data",
    "has session": "Session",
    "has observation": "Observation",
    "has timepoint": "Timepoint",
    "measures parameter": "Parameter",
    "has quality assessment": "Data Quality Assessment",
    "produced by sensor": "Sensor",
    "has injury mechanism": "Demographic Data",
}

_RELATION_DOMAIN_SUPERCLASS: Dict[str, str] = {
    "monitoring indicates condition": "Monitoring Data",
    "targets condition": "Therapy",
    "triggers intervention": "Condition",
    "has observation": "Timepoint",
    "has timepoint": "Session",
    "measures parameter": "Observation",
    "produced by sensor": "Observation",
    "has quality assessment": "Observation",
    "affected by treatment": "Data Quality Assessment",
    "associated with condition": "Data Quality Assessment",
}

_RANGE_SUPERCLASS_SUBTREE: Dict[str, Set[str]] = {}


def _build_superclass_subtree(ontology: Ontology) -> Dict[str, Set[str]]:
    """Build a mapping from each abstract range superclass to the set of all its
    transitive subclass canonical keys (including itself), using the ontology hierarchy."""
    parent_to_children: Dict[str, Set[str]] = {}
    for e in ontology.hierarchy:
        sub = canonical_key(resolve_to_canonical_label((e.get("subClass") or "").strip())[0])
        sup = canonical_key(resolve_to_canonical_label((e.get("superClass") or "").strip())[0])
        if sub and sup and sub != sup:
            parent_to_children.setdefault(sup, set()).add(sub)

    result: Dict[str, Set[str]] = {}
    for superclass_label in _RELATION_RANGE_SUPERCLASS.values():
        root_key = canonical_key(resolve_to_canonical_label(superclass_label)[0])
        descendants: Set[str] = {root_key}
        frontier = [root_key]
        while frontier:
            cur = frontier.pop()
            for child in parent_to_children.get(cur, set()):
                if child not in descendants:
                    descendants.add(child)
                    frontier.append(child)
        result[root_key] = descendants
    return result


def _normalize_relation_ranges(ontology: Ontology) -> int:
    """Rewrite relation domains/ranges from specific subclasses to correct abstract superclasses.

    Range normalization: e.g. has monitoring data(Patient → EtCO2) → (Patient → Monitoring Data).
    Domain normalization: e.g. targets condition(Vasopressors → Condition) → (Therapy → Condition).
    Returns count of relations where domain or range was normalized."""
    subtree = _build_superclass_subtree(ontology)
    normalized = 0

    domain_subtree: Dict[str, Set[str]] = {}
    for domain_label in _RELATION_DOMAIN_SUPERCLASS.values():
        root_key = canonical_key(resolve_to_canonical_label(domain_label)[0])
        if root_key not in subtree:
            parent_to_children: Dict[str, Set[str]] = {}
            for e in ontology.hierarchy:
                sub = canonical_key(resolve_to_canonical_label((e.get("subClass") or "").strip())[0])
                sup = canonical_key(resolve_to_canonical_label((e.get("superClass") or "").strip())[0])
                if sub and sup and sub != sup:
                    parent_to_children.setdefault(sup, set()).add(sub)
            descendants: Set[str] = {root_key}
            frontier = [root_key]
            while frontier:
                cur = frontier.pop()
                for child in parent_to_children.get(cur, set()):
                    if child not in descendants:
                        descendants.add(child)
                        frontier.append(child)
            domain_subtree[root_key] = descendants
        else:
            domain_subtree[root_key] = subtree[root_key]

    for r in ontology.relations:
        rel_norm = _norm((r.label or "").strip())
        changed = False

        expected_range = None
        for rel_pattern, range_label in _RELATION_RANGE_SUPERCLASS.items():
            if _norm(rel_pattern) == rel_norm:
                expected_range = range_label
                break

        if expected_range is not None:
            current_range = (r.range or "").strip()
            range_key = canonical_key(resolve_to_canonical_label(current_range)[0])
            expected_key = canonical_key(resolve_to_canonical_label(expected_range)[0])
            if range_key != expected_key:
                descendants = subtree.get(expected_key, set())
                if range_key in descendants:
                    r.range = expected_range
                    changed = True

        expected_domain = None
        for rel_pattern, domain_label in _RELATION_DOMAIN_SUPERCLASS.items():
            if _norm(rel_pattern) == rel_norm:
                expected_domain = domain_label
                break

        if expected_domain is not None:
            current_domain = (r.domain or "").strip()
            domain_key = canonical_key(resolve_to_canonical_label(current_domain)[0])
            expected_dom_key = canonical_key(resolve_to_canonical_label(expected_domain)[0])
            if domain_key != expected_dom_key:
                dom_descendants = domain_subtree.get(expected_dom_key, set())
                if domain_key in dom_descendants:
                    r.domain = expected_domain
                    changed = True

        if changed:
            normalized += 1

    return normalized


def _dedupe_relations(ontology: Ontology, gold_label_by_norm: Dict[str, str]) -> None:
    seen: Set[tuple] = set()
    kept: List[RelationEntity] = []
    for r in ontology.relations:
        label_k = canonical_key(r.label or "")
        dom_res = resolve_to_canonical_label(r.domain or "")[0] if r.domain else ""
        ran_res = resolve_to_canonical_label(r.range or "")[0] if r.range else ""
        dom_k = canonical_key(dom_res)
        ran_k = canonical_key(ran_res)
        if not label_k or not dom_k or not ran_k:
            continue
        key = (label_k, dom_k, ran_k)
        if key in seen:
            continue
        seen.add(key)
        domain = gold_label_by_norm.get(canonical_key(dom_res), r.domain) if r.domain else None
        range_ = gold_label_by_norm.get(canonical_key(ran_res), r.range) if r.range else None
        kept.append(
            RelationEntity(
                label=r.label,
                domain=domain,
                range=range_,
                definition=r.definition,
                provenance=list(r.provenance) + ["symbolic_reasoner_dedupe"],
                evidence=getattr(r, "evidence", None),
                aliases=list(getattr(r, "aliases", None) or []),
                stratum=getattr(r, "stratum", None),
            )
        )
    ontology.relations.clear()
    ontology.relations.extend(kept)


def _dedupe_hierarchy(ontology: Ontology, gold_label_by_norm: Dict[str, str]) -> None:
    seen: Set[tuple] = set()
    kept: List[Dict] = []
    for e in ontology.hierarchy:
        sub = e.get("subClass", "").strip()
        sup = e.get("superClass", "").strip()
        if not sub or not sup:
            continue
        sub_res = resolve_to_canonical_label(sub)[0]
        sup_res = resolve_to_canonical_label(sup)[0]
        sub_k = canonical_key(sub_res)
        sup_k = canonical_key(sup_res)
        if sub_k == sup_k:
            continue
        key = (sub_k, sup_k)
        if key in seen:
            continue
        seen.add(key)
        sub_canon = gold_label_by_norm.get(sub_k, sub_res)
        sup_canon = gold_label_by_norm.get(sup_k, sup_res)
        kept.append({"subClass": sub_canon, "superClass": sup_canon, **{k: v for k, v in e.items() if k not in ("subClass", "superClass")}})
    ontology.hierarchy.clear()
    ontology.hierarchy.extend(kept)


def _remove_hierarchy_cycles(ontology: Ontology, *, audit_log: Optional[List] = None) -> int:
    """Detect and remove edges that participate in cycles (A ⊑ B ⊑ … ⊑ A).

    Uses Kahn's algorithm: edges remaining after topological sort removal
    are part of cycles and are discarded. Returns number of edges removed.
    """
    edges = ontology.hierarchy
    if not edges:
        return 0

    child_to_parents: Dict[str, Set[str]] = {}
    parent_to_children: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set()
    edge_keys: Dict[tuple, Dict] = {}

    for e in edges:
        sub = canonical_key(resolve_to_canonical_label((e.get("subClass") or "").strip())[0])
        sup = canonical_key(resolve_to_canonical_label((e.get("superClass") or "").strip())[0])
        if not sub or not sup or sub == sup:
            continue
        child_to_parents.setdefault(sub, set()).add(sup)
        parent_to_children.setdefault(sup, set()).add(sub)
        all_nodes.update((sub, sup))
        edge_keys[(sub, sup)] = e

    in_degree = {n: len(child_to_parents.get(n, set())) for n in all_nodes}
    roots = [n for n, d in in_degree.items() if d == 0]
    visited: Set[str] = set()
    frontier = list(roots)

    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        for child in parent_to_children.get(node, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                frontier.append(child)

    cycle_nodes = all_nodes - visited
    if not cycle_nodes:
        return 0

    kept: List[Dict] = []
    removed = 0
    for e in edges:
        sub_raw = (e.get("subClass") or "").strip()
        sup_raw = (e.get("superClass") or "").strip()
        sub = canonical_key(resolve_to_canonical_label(sub_raw)[0])
        sup = canonical_key(resolve_to_canonical_label(sup_raw)[0])
        if sub in cycle_nodes and sup in cycle_nodes:
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "hierarchy", "subClass": sub_raw, "superClass": sup_raw, "stage": "cycle_removal", "reason": "edge participates in a hierarchy cycle"})
        else:
            kept.append(e)

    ontology.hierarchy.clear()
    ontology.hierarchy.extend(kept)
    return removed


def _build_remap_for_endpoints(
    ontology: Ontology, allowed_class_norm: Set[str]
) -> Dict[str, str]:
    """Build a lookup from canonical key of surviving classes to their display label.
    Used for soft-remapping relation endpoints and hierarchy nodes before pruning."""
    norm_to_label: Dict[str, str] = {}
    for c in ontology.classes:
        k = canonical_key(resolve_to_canonical_label(c.label or "")[0])
        if k and k in allowed_class_norm:
            norm_to_label[k] = c.label
    return norm_to_label


def _try_remap_endpoint(
    label: str,
    allowed_class_norm: Set[str],
    norm_to_label: Dict[str, str],
) -> str | None:
    """Try to remap a relation endpoint to a surviving class. Returns the surviving label or None."""
    if not label:
        return None
    resolved = resolve_to_canonical_label(label)[0]
    k = canonical_key(resolved)
    if k in allowed_class_norm:
        return norm_to_label.get(k, label)
    try:
        from ..prompting.vocabulary import CLASS_SYNONYM_MAP
        lower_map = {mk.lower(): mv for mk, mv in CLASS_SYNONYM_MAP.items() if mv is not None}
        remapped = lower_map.get(label.lower().strip())
        if remapped:
            rk = canonical_key(resolve_to_canonical_label(remapped)[0])
            if rk in allowed_class_norm:
                return norm_to_label.get(rk, remapped)
    except ImportError:
        pass
    return None


def _auto_add_missing_endpoints(
    ontology: Ontology,
    surviving_class_norm: Set[str],
    *,
    removed_class_keys: Optional[Set[str]] = None,
) -> int:
    """Auto-add gold-vocabulary classes that are referenced as relation/hierarchy
    endpoints but were not extracted as classes. This prevents the cascade wipe
    that occurs when e.g. 'Patient' is used in every relation domain but was
    never listed as a class by the LLM.

    Classes whose keys appear in *removed_class_keys* (i.e. explicitly removed
    by earlier cleanup stages) are never re-added.
    """
    from ..prompting.vocabulary import ALLOWED_CLASSES_CORE
    from ..ontology.model import ClassEntity

    blocked = removed_class_keys or set()

    allowed_norm_map: Dict[str, str] = {}
    for lbl in ALLOWED_CLASSES_CORE:
        k = canonical_key(resolve_to_canonical_label(lbl)[0])
        if k:
            allowed_norm_map[k] = lbl

    referenced: Set[str] = set()
    for r in ontology.relations:
        for ep in (r.domain, r.range):
            if ep:
                referenced.add(ep.strip())
    for h in ontology.hierarchy:
        for ep in (h.get("subClass", ""), h.get("superClass", "")):
            if ep and ep.strip():
                referenced.add(ep.strip())

    added = 0
    for ep in referenced:
        k = canonical_key(resolve_to_canonical_label(ep)[0])
        if k in surviving_class_norm:
            continue
        if k in blocked:
            continue
        if k not in allowed_norm_map:
            continue
        canonical_label = allowed_norm_map[k]
        ontology.classes.append(ClassEntity(
            label=canonical_label,
            evidence=f"Implicit: referenced as relation/hierarchy endpoint.",
            provenance=["auto_endpoint_recovery"],
        ))
        surviving_class_norm.add(k)
        added += 1

    return added


def _auto_add_missing_endpoints_gold_free(
    ontology: Ontology,
    surviving_class_norm: Set[str],
    *,
    removed_class_keys: Optional[Set[str]] = None,
) -> int:
    """Gold-free version: auto-add classes referenced as relation/hierarchy endpoints
    but not in the class list. Unlike the gold version, this does NOT consult
    ALLOWED_CLASSES_CORE — it simply adds the endpoint label as-is. This prevents
    cascade deletion of relations when e.g. 'Patient' is used as domain everywhere
    but wasn't extracted as a class.

    Classes whose keys appear in *removed_class_keys* (i.e. explicitly removed
    by earlier cleanup stages) are never re-added.
    """
    from ..ontology.model import ClassEntity

    blocked = removed_class_keys or set()

    referenced: Set[str] = set()
    for r in ontology.relations:
        for ep in (r.domain, r.range):
            if ep:
                referenced.add(ep.strip())
    for h in ontology.hierarchy:
        for ep in (h.get("subClass", ""), h.get("superClass", "")):
            if ep and ep.strip():
                referenced.add(ep.strip())

    added = 0
    for ep in referenced:
        k = canonical_key(resolve_to_canonical_label(ep)[0])
        if k in surviving_class_norm:
            continue
        if k in blocked:
            continue
        ontology.classes.append(ClassEntity(
            label=ep,
            evidence="Implicit: referenced as relation/hierarchy endpoint.",
            provenance=["auto_endpoint_recovery"],
        ))
        surviving_class_norm.add(k)
        added += 1

    return added


def _correct_flat_hierarchy(ontology: Ontology) -> int:
    """Correct flat hierarchy mappings where the LLM skipped intermediate levels.

    Example: Heart Rate ⊑ Monitoring Data → Heart Rate ⊑ Core Monitoring Parameter
    (when Core Monitoring Parameter exists in the ontology).

    Only corrects if the correct parent class actually exists in the ontology."""
    existing_class_norms = {_norm(c.label or "") for c in ontology.classes if c.label}
    corrected = 0

    for edge in ontology.hierarchy:
        sub = (edge.get("subClass") or "").strip()
        if not sub:
            continue
        sub_norm = _norm(sub)
        correction = _HIERARCHY_CORRECT_PARENT_NORM.get(sub_norm)
        if correction is None:
            continue
        correct_child_label, correct_parent_label = correction
        correct_parent_norm = _norm(correct_parent_label)
        current_super = (edge.get("superClass") or "").strip()
        current_super_norm = _norm(current_super)

        if current_super_norm == correct_parent_norm:
            continue
        if correct_parent_norm not in existing_class_norms:
            continue

        edge["superClass"] = correct_parent_label
        corrected += 1

    return corrected


def _scaffold_hierarchy_from_domain_scope(
    ontology: Ontology,
    *,
    audit_log: Optional[List] = None,
) -> int:
    """Phase 27.4: Auto-generate hierarchy edges from domain_scope.json categories.

    For each (category → member) pair defined in concept_categories, if both the
    category parent and the member class exist in the ontology AND no hierarchy
    edge already connects them, add an inferred edge.

    Uses _HIERARCHY_CORRECT_PARENT as the authoritative mapping (child → correct
    immediate parent). This is more precise than the broad concept_categories
    grouping because it respects intermediate levels (e.g. Heart Rate → Core
    Monitoring Parameter, not Monitoring Data directly).

    Returns the number of edges added.
    """
    existing_class_norms = {_norm(c.label or "") for c in ontology.classes if c.label}
    existing_class_key_to_label: Dict[str, str] = {}
    for c in ontology.classes:
        if c.label:
            existing_class_key_to_label[canonical_key(resolve_to_canonical_label(c.label)[0])] = c.label

    existing_hierarchy_keys: Set[tuple] = set()
    for e in ontology.hierarchy:
        sub = canonical_key(resolve_to_canonical_label((e.get("subClass") or "").strip())[0])
        sup = canonical_key(resolve_to_canonical_label((e.get("superClass") or "").strip())[0])
        if sub and sup:
            existing_hierarchy_keys.add((sub, sup))

    added = 0

    # Use the authoritative child → parent mapping
    for child_label, parent_label in _HIERARCHY_CORRECT_PARENT.items():
        child_norm = _norm(child_label)
        parent_norm = _norm(parent_label)
        if child_norm not in existing_class_norms or parent_norm not in existing_class_norms:
            continue

        child_key = canonical_key(resolve_to_canonical_label(child_label)[0])
        parent_key = canonical_key(resolve_to_canonical_label(parent_label)[0])
        if not child_key or not parent_key:
            continue
        if child_key == parent_key:
            continue
        if (child_key, parent_key) in existing_hierarchy_keys:
            continue

        # Find actual labels in the ontology
        actual_child = existing_class_key_to_label.get(child_key, child_label)
        actual_parent = existing_class_key_to_label.get(parent_key, parent_label)

        ontology.hierarchy.append({
            "subClass": actual_child,
            "superClass": actual_parent,
            "evidence": f"Domain-scope scaffolding: {actual_child} is a known subclass of {actual_parent} in the BrainIT domain.",
            "provenance": ["domain_scope_scaffolding"],
            "stratum": "inferred",
        })
        existing_hierarchy_keys.add((child_key, parent_key))
        added += 1

        if audit_log is not None:
            audit_log.append({
                "type": "hierarchy",
                "subClass": actual_child,
                "superClass": actual_parent,
                "stage": "domain_scope_scaffolding",
                "reason": f"Inferred from domain knowledge: {actual_child} subClassOf {actual_parent}",
            })

    return added


def _log_hierarchy_completeness(ontology: Ontology) -> Dict:
    """Phase 27.5: Log hierarchy completeness metrics after all cleanup.

    Returns a dict with orphan_count, orphan_rate, and a warning flag.
    """
    class_labels = {(c.label or "").strip() for c in ontology.classes if c.label}
    children = set()
    for e in ontology.hierarchy:
        sub = (e.get("subClass") or "").strip()
        if sub:
            children.add(sub)

    # A class is an orphan if it has no parent in the hierarchy
    orphans = class_labels - children
    total = len(class_labels) or 1
    orphan_rate = len(orphans) / total

    return {
        "orphan_count": len(orphans),
        "total_classes": len(class_labels),
        "orphan_rate": round(orphan_rate, 3),
        "orphan_rate_warning": orphan_rate > 0.5,
    }


def _prune_relations_domain_range(ontology: Ontology, allowed_class_norm: Set[str], *, audit_log: Optional[List] = None) -> None:
    """Remove relations whose domain or range is not in the allowed (schema) class set.
    Before deleting, try to remap endpoints to surviving classes via synonym resolution."""
    norm_to_label = _build_remap_for_endpoints(ontology, allowed_class_norm)
    kept = []
    for r in ontology.relations:
        dom = r.domain
        ran = r.range
        if dom:
            remapped_dom = _try_remap_endpoint(dom, allowed_class_norm, norm_to_label)
            if remapped_dom is None:
                if audit_log is not None:
                    audit_log.append({"type": "relation", "label": r.label, "domain": dom, "range": ran, "stage": "dangling_domain", "reason": f"domain '{dom}' not in surviving classes and could not be remapped"})
                continue
            r.domain = remapped_dom
        if ran:
            remapped_ran = _try_remap_endpoint(ran, allowed_class_norm, norm_to_label)
            if remapped_ran is None:
                if audit_log is not None:
                    audit_log.append({"type": "relation", "label": r.label, "domain": dom, "range": ran, "stage": "dangling_range", "reason": f"range '{ran}' not in surviving classes and could not be remapped"})
                continue
            r.range = remapped_ran
        kept.append(r)
    ontology.relations.clear()
    ontology.relations.extend(kept)


def _prune_hierarchy_dangling_endpoints(ontology: Ontology, allowed_class_norm: Set[str], *, audit_log: Optional[List] = None) -> None:
    """Remove hierarchy edges whose subClass or superClass was pruned (not in surviving class set).
    Before deleting, try to remap endpoints to surviving classes via synonym resolution."""
    norm_to_label = _build_remap_for_endpoints(ontology, allowed_class_norm)
    kept = []
    for e in ontology.hierarchy:
        sub = (e.get("subClass") or "").strip()
        sup = (e.get("superClass") or "").strip()
        if not sub or not sup:
            if audit_log is not None:
                audit_log.append({"type": "hierarchy", "subClass": sub, "superClass": sup, "stage": "dangling_hierarchy", "reason": "empty subClass or superClass"})
            continue
        remapped_sub = _try_remap_endpoint(sub, allowed_class_norm, norm_to_label)
        remapped_sup = _try_remap_endpoint(sup, allowed_class_norm, norm_to_label)
        if remapped_sub is None or remapped_sup is None:
            if audit_log is not None:
                failed = []
                if remapped_sub is None: failed.append(f"subClass '{sub}'")
                if remapped_sup is None: failed.append(f"superClass '{sup}'")
                audit_log.append({"type": "hierarchy", "subClass": sub, "superClass": sup, "stage": "dangling_hierarchy", "reason": f"{' and '.join(failed)} not in surviving classes"})
            continue
        if remapped_sub.lower() == remapped_sup.lower():
            if audit_log is not None:
                audit_log.append({"type": "hierarchy", "subClass": sub, "superClass": sup, "stage": "dangling_hierarchy", "reason": f"self-loop after remapping (both resolve to '{remapped_sub}')"})
            continue
        e["subClass"] = remapped_sub
        e["superClass"] = remapped_sup
        kept.append(e)
    ontology.hierarchy.clear()
    ontology.hierarchy.extend(kept)


def _prune_out_of_scope_classes(ontology: Ontology, *, audit_log: Optional[List] = None) -> int:
    """Remove classes whose label or evidence indicates governance/org/admin/database context.

    Also removes classes listed in domain_scope.json out_of_scope_classes (Phase 29).
    """
    # Phase 29: Load explicit out-of-scope class list from domain scope
    _oos_norm: Set[str] = set()
    try:
        from .domain_scope import get_out_of_scope_classes_norm
        _oos_norm = set(get_out_of_scope_classes_norm())
    except Exception:
        pass

    kept: List[ClassEntity] = []
    removed = 0
    for c in ontology.classes:
        label = (c.label or "").strip()
        label_lower = label.lower()
        evidence = (getattr(c, "evidence", None) or "").lower()

        # Phase 29: Check explicit out-of-scope list (normalized comparison)
        if _oos_norm and _norm(label) in _oos_norm:
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "class", "label": c.label, "stage": "out_of_scope", "reason": "label in domain_scope out_of_scope_classes list"})
            continue

        if any(re.search(p, label_lower) for p in _OUT_OF_SCOPE_CLASS_PATTERNS):
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "class", "label": c.label, "stage": "out_of_scope", "reason": "label matches out-of-scope pattern"})
            continue
        if evidence and any(re.search(p, evidence) for p in _OUT_OF_SCOPE_EVIDENCE_PATTERNS):
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "class", "label": c.label, "stage": "out_of_scope", "reason": "evidence matches out-of-scope pattern"})
            continue
        kept.append(c)
    ontology.classes.clear()
    ontology.classes.extend(kept)
    return removed


def _prune_abstract_data_labels(ontology: Ontology, *, audit_log: Optional[List] = None) -> int:
    """Remove classes matching abstract/data patterns unless in allowlist."""
    _allowlist = _get_abstract_label_allowlist()
    kept: List[ClassEntity] = []
    removed = 0
    for c in ontology.classes:
        label = (c.label or "").strip()
        label_norm = _norm_label(label)
        if label_norm in _allowlist:
            kept.append(c)
            continue
        if not any(re.search(p, label.lower()) for p in _ABSTRACT_DATA_LABEL_PATTERNS):
            kept.append(c)
            continue
        removed += 1
        if audit_log is not None:
            audit_log.append({"type": "class", "label": c.label, "stage": "abstract_data", "reason": "label matches abstract/data pattern"})
    ontology.classes.clear()
    ontology.classes.extend(kept)
    return removed


def _prune_broad_contextual_classes(ontology: Ontology, *, audit_log: Optional[List] = None) -> int:
    """Remove classes with broad contextual labels (e.g. head injured patients, new therapies, monitoring devices) unless in allowlist."""
    kept: List[ClassEntity] = []
    removed = 0
    for c in ontology.classes:
        lab = (c.label or "").strip().lower()
        label_norm = _norm_label(c.label or "")
        if label_norm in _get_broad_context_allowlist():
            kept.append(c)
            continue
        if any(re.search(p, lab) for p in _BROAD_CONTEXT_LABEL_PATTERNS):
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "class", "label": c.label, "stage": "broad_contextual", "reason": "label matches broad contextual pattern"})
            continue
        kept.append(c)
    ontology.classes.clear()
    ontology.classes.extend(kept)
    return removed


def _is_out_of_scope_label(label: str) -> bool:
    """True if a class label matches any out-of-scope pattern."""
    lab_lower = label.strip().lower()
    return any(re.search(p, lab_lower) for p in _OUT_OF_SCOPE_CLASS_PATTERNS)


def _prune_edges_with_out_of_scope_endpoints(ontology, *, audit_log: Optional[List] = None) -> None:
    """Remove relations and hierarchy edges whose domain/range/sub/super match out-of-scope patterns.

    This prevents _auto_add_missing_endpoints from re-introducing noise classes
    that were already pruned by _prune_out_of_scope_classes.
    """
    kept_rels = []
    for r in ontology.relations:
        dom_bad = _is_out_of_scope_label(r.domain or "")
        ran_bad = _is_out_of_scope_label(r.range or "")
        if dom_bad or ran_bad:
            if audit_log is not None:
                audit_log.append({"type": "relation", "label": r.label, "domain": r.domain, "range": r.range, "stage": "scope_endpoint", "reason": f"endpoint out-of-scope: domain={dom_bad}, range={ran_bad}"})
        else:
            kept_rels.append(r)
    ontology.relations[:] = kept_rels

    kept_hier = []
    for h in ontology.hierarchy:
        sub_bad = _is_out_of_scope_label(h.get("subClass", ""))
        sup_bad = _is_out_of_scope_label(h.get("superClass", ""))
        if sub_bad or sup_bad:
            if audit_log is not None:
                audit_log.append({"type": "hierarchy", "subClass": h.get("subClass"), "superClass": h.get("superClass"), "stage": "scope_endpoint", "reason": f"endpoint out-of-scope: sub={sub_bad}, super={sup_bad}"})
        else:
            kept_hier.append(h)
    ontology.hierarchy[:] = kept_hier


def _label_evidenced_in_evidence(c: ClassEntity, ev: str) -> bool:
    """
    True if evidence supports this class.

    Matching pipeline (in order — returns True on first hit):
    1. Exact normalised substring match.
    2. Alias match (class aliases list).
    3. Reverse synonym map lookup.
    4. Parenthetical abbreviation extraction.
    5. Raw token overlap ≥ threshold.
    6. Short abbreviation expansion via corpus vocab.
    7. Corpus synonym group lookup.
    8. [S1] Lemmatised token overlap (morphological variants: monitor/monitoring).
    9. [S2] spaCy semantic similarity ≥ threshold (bridges pulse oximetry ↔ oxygen saturation).
    """
    if not ev or len(ev) < _CLASS_EVIDENCE_MIN_LENGTH:
        return False
    ev_norm = _norm(ev)
    ev_lower = (ev or "").lower()
    lab_norm = _norm_label(c.label or "")

    # 1. Exact normalised substring
    if lab_norm and lab_norm in ev_norm:
        return True

    # 2. Alias match
    aliases = getattr(c, "aliases", None) or []
    for a in aliases:
        if _norm(a) in ev_norm:
            return True

    # 3. Reverse synonym map
    label_lower = (c.label or "").strip().lower()
    for syn in _REVERSE_SYNONYM_MAP.get(label_lower, []):
        if _norm(syn) in ev_norm:
            return True
        if syn.lower() in ev_lower:
            return True

    # 4. Parenthetical abbreviation
    _lbl = c.label or ""
    _open = _lbl.find("(")
    _close = _lbl.find(")", _open + 1 if _open >= 0 else 0)
    if _open >= 0 and _close > _open:
        inside = _lbl[_open + 1 : _close].strip()
        if inside and _norm(inside) in ev_norm:
            return True

    # 5. Raw token overlap
    lab_tokens = _tokens(c.label or "")
    ev_tokens = _tokens(ev)
    if _token_overlap_ratio(lab_tokens, ev_tokens) >= _CLASS_EVIDENCE_TOKEN_OVERLAP_THRESHOLD:
        return True

    # 6. Short abbreviation expansion via corpus vocab
    clean_label = (c.label or "").strip()
    if len(clean_label) <= 7 and clean_label == clean_label.upper() and _corpus_dict_loaded():
        expansion = _corpus_get_abbr_expansion(clean_label)
        if expansion:
            exp_norm = _norm(expansion)
            if exp_norm and exp_norm in ev_norm:
                return True
            exp_tokens = _tokens(expansion)
            if exp_tokens and _token_overlap_ratio(exp_tokens, ev_tokens) >= _CLASS_EVIDENCE_TOKEN_OVERLAP_THRESHOLD:
                return True

    # 7. Corpus synonym group lookup
    if _corpus_dict_loaded():
        syn_variants = _corpus_get_synonym_group(c.label or "")
        for variant in syn_variants:
            vn = _norm(variant)
            if vn and vn in ev_norm:
                return True

    # ── Strategies 8 & 9 require spaCy (graceful fallback if unavailable) ──────
    _nlp = _get_sci_nlp()
    if _nlp is None:
        return False

    # 8. [S1] Lemmatised token overlap — catches morphological variants
    # e.g. "monitoring" ↔ "monitor", "arterial" ↔ "artery"
    try:
        lab_lemmas = _lemma_set(c.label or "", _nlp)
        ev_lemmas = _lemma_set(ev, _nlp)
        if lab_lemmas and len(lab_lemmas & ev_lemmas) >= max(1, len(lab_lemmas) // 2):
            return True
    except Exception:
        pass

    # 9. [S2] Max word-pair semantic similarity — catches clinical synonyms with no
    # lexical overlap, e.g. "Arterial Oxygen Saturation" ↔ "pulse oximetry" (0.72).
    # Uses the best single (label_token, evidence_token) cosine pair, which is
    # more robust than doc-level similarity which gets diluted by non-matching words.
    try:
        if _max_word_pair_similarity(clean_label, ev, _nlp) >= _EVIDENCE_SIMILARITY_THRESHOLD:
            return True
    except Exception:
        pass

    return False


def _prune_classes_by_evidence(ontology: Ontology, *, gold_free: bool = False, audit_log: Optional[List] = None) -> int:
    """Remove classes with missing/short evidence or where label is not supported (normalized/alias/token overlap).

    Evidence exemptions and in-scope classes are loaded from the domain scope file
    (resources/domain_scope.json). In gold_free mode, structural exemptions from
    domain scope still apply. In Guided/Schema modes, the full domain scope
    exemptions + in-scope class list are used.
    """
    domain_exemptions = set(_get_evidence_pruning_exemptions())
    domain_exemptions |= set(_get_structural_evidence_exemptions())
    if not gold_free:
        try:
            from .domain_scope import get_in_scope_classes_norm
            domain_exemptions |= set(get_in_scope_classes_norm())
        except Exception:
            from ..prompting.vocabulary import ALLOWED_CLASSES_CORE
            for lbl in ALLOWED_CLASSES_CORE:
                domain_exemptions.add(_norm_label(lbl))
    gold_norms: Set[str] = domain_exemptions

    kept: List[ClassEntity] = []
    removed = 0
    for c in ontology.classes:
        ev_str = getattr(c, "evidence", None) or ""
        if "chain-of-layer" in ev_str.lower() or "intermediate category" in ev_str.lower():
            kept.append(c)
            continue
        if _norm_label(c.label or "") in gold_norms:
            kept.append(c)
            continue
        ev = (getattr(c, "evidence", None) or "")
        if not _label_evidenced_in_evidence(c, ev):
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "class", "label": c.label, "stage": "evidence", "reason": "label not evidenced in evidence text", "evidence_snippet": ev[:200] if ev else ""})
            continue
        kept.append(c)
    ontology.classes.clear()
    ontology.classes.extend(kept)
    return removed


def _evidence_contains_class_reference(
    ev_norm: str,
    class_label: str,
    norm_to_aliases: Dict[str, List[str]],
    ev_raw: str = "",
) -> bool:
    """True if evidence (normalised) contains the class label or any of its aliases.

    Matching strategies (in order):
    1. Exact normalised substring.
    2. Alias match from class aliases or corpus abbreviation expansions.
    3. Parenthetical extraction (e.g. CPP from 'Cerebral Perfusion Pressure (CPP)').
    4. Corpus abbreviation expansion (e.g. ICP → 'intracranial pressure').
    5. Token overlap: 2+ significant tokens from label appear in evidence.
    6. Single significant token match for short labels.
    7. [S1] Lemmatised token overlap (morphological variants).
    8. [S2] spaCy semantic similarity (clinical synonyms with no lexical overlap).
    """
    if not class_label or not ev_norm:
        return False
    cn = _norm(class_label)

    # 1. Exact normalised substring
    if cn and cn in ev_norm:
        return True

    # 2. Alias match
    for alias_norm in norm_to_aliases.get(cn, []):
        if alias_norm and alias_norm in ev_norm:
            return True

    # 3. Parenthetical abbreviation
    _op = class_label.find("(")
    _cl = class_label.find(")", _op + 1 if _op >= 0 else 0)
    if _op >= 0 and _cl > _op:
        inside = class_label[_op + 1 : _cl].strip()
        if inside:
            inside_norm = _norm(inside)
            if inside_norm and inside_norm in ev_norm:
                return True

    # 4. Corpus abbreviation expansion
    clean_label = class_label.strip()
    if _corpus_dict_loaded():
        expansion = _corpus_get_abbr_expansion(clean_label)
        if expansion:
            exp_norm = _norm(expansion)
            if exp_norm and exp_norm in ev_norm:
                return True
        from ..corpus.corpus_vocab import get_reverse_abbreviation
        rev_abbr = get_reverse_abbreviation(clean_label)
        if rev_abbr:
            rev_norm = _norm(rev_abbr)
            if rev_norm and rev_norm in ev_norm:
                return True

    # 5. Token overlap: 2+ significant tokens from label appear in evidence
    tokens = [t for t in re.split(r"[^a-z0-9]+", (class_label or "").lower()) if len(t) >= 2]
    if len(tokens) >= 2:
        matches = sum(1 for t in tokens if t and t in ev_norm)
        if matches >= 2:
            return True

    # 6. Single significant token for short labels
    if len(tokens) == 1 and len(tokens[0]) >= 4:
        if tokens[0] in ev_norm:
            return True

    # ── Strategies 7 & 8 require spaCy ─────────────────────────────────────────
    _nlp = _get_sci_nlp()
    if _nlp is None:
        return False

    # 7. [S1] Lemmatised token overlap
    try:
        lab_lemmas = _lemma_set(class_label, _nlp)
        ev_text = ev_raw or ev_norm
        ev_lemmas = _lemma_set(ev_text, _nlp)
        if lab_lemmas and len(lab_lemmas & ev_lemmas) >= max(1, len(lab_lemmas) // 2):
            return True
    except Exception:
        pass

    # 8. [S2] Max word-pair semantic similarity
    try:
        ev_text = ev_raw or ev_norm
        if _max_word_pair_similarity(clean_label, ev_text, _nlp) >= _EVIDENCE_SIMILARITY_THRESHOLD:
            return True
    except Exception:
        pass

    return False


def _normalize_relation_label(label: str) -> str:
    """Normalize an LLM-generated relation label to a canonical form using
    the normalization map. Returns the canonical label if found, else the original."""
    lab_norm = _norm(label)
    for variant, canonical in _RELATION_LABEL_NORMALIZATION.items():
        if _norm(variant) == lab_norm:
            return canonical
    return label


_RELATION_LABEL_BLOCKLIST = frozenset({
    "data", "information", "value", "type", "class", "category",
    "entity", "concept", "thing", "item", "element", "result",
    "study", "paper", "analysis", "method", "approach", "system",
    "group", "project", "model", "process", "network", "framework",
    # Phase 26: Block statistical/methodology relations
    "computed as correlation", "has regression model", "has regression",
    "has correlation", "has statistical model", "has prediction model",
    "has p value", "has confidence interval",
})


def _is_valid_relation_label(label: str) -> bool:
    """True if *label* looks like a plausible relation name.

    Rejects: empty, too-long (>6 words — likely a sentence), purely
    numeric, purely generic nouns that are class-like rather than
    relational, or strings that look like full evidence fragments.
    """
    s = (label or "").strip()
    if not s or len(s) < 2:
        return False
    if not any(c.isalpha() for c in s):
        return False
    words = s.split()
    if len(words) > 6:
        return False
    if s.lower() in _RELATION_LABEL_BLOCKLIST:
        return False
    return True


def _prune_weak_relations_by_evidence(
    ontology: Ontology,
    gold_relation_labels_norm: Set[str] | None = None,
    *,
    audit_log: Optional[List] = None,
) -> int:
    """
    Remove relations that lack evidence, have invalid labels, or whose
    evidence does not mention either endpoint.

    Label filtering uses a **blocklist + quality-check** approach rather
    than a strict whitelist.  Any label that looks like a plausible
    relation name (1–6 words, not a generic noun) is accepted.  Labels
    that match ``_ALLOWED_RELATION_LABELS_GLOBAL`` are always accepted;
    labels that pass ``_is_valid_relation_label`` are also accepted.

    Before the label check, each label is normalised through
    ``_RELATION_LABEL_NORMALIZATION`` to map common LLM variants to
    canonical forms.

    Exemption tiers for the endpoint-evidence check:
    1. Schema-guided relations (stratum="schema_guided"): already corpus-validated.
    2. Gold-schema-licensed relations: label matches a gold schema relation label.
    3. TGC-sourced relations: produced by text-grounded completion with corpus evidence.
    """
    allowed_norm = {_norm(x) for x in _ALLOWED_RELATION_LABELS_GLOBAL}
    _gold_labels: Set[str] = gold_relation_labels_norm or set()
    norm_to_aliases: Dict[str, List[str]] = {}
    for c in ontology.classes:
        cn = _norm_label(c.label or "")
        if not cn:
            continue
        aliases = getattr(c, "aliases", None) or []
        norm_to_aliases[cn] = [_norm(a) for a in aliases if _norm(a)]

    if _corpus_dict_loaded():
        from ..corpus.corpus_vocab import get_all_abbreviations
        for abbr, expansion in get_all_abbreviations().items():
            abbr_norm = _norm(abbr)
            exp_norm = _norm(expansion)
            if abbr_norm and exp_norm:
                norm_to_aliases.setdefault(abbr_norm, []).append(exp_norm)
                norm_to_aliases.setdefault(exp_norm, []).append(abbr_norm)

    kept: List[RelationEntity] = []
    removed = 0
    for r in ontology.relations:
        ev = (getattr(r, "evidence", None) or "")
        ev_norm = _norm(ev)
        dom = (r.domain or "").strip()
        ran = (r.range or "").strip()
        lab = (r.label or "").strip()

        normalized_lab = _normalize_relation_label(lab)
        if normalized_lab != lab:
            r.label = normalized_lab
            lab = normalized_lab
        lab_norm = _norm(lab)

        is_schema_guided = getattr(r, "stratum", None) == "schema_guided"
        is_gold_licensed = lab_norm in _gold_labels
        is_tgc = getattr(r, "stratum", None) in ("tgc", "text_grounded_completion")

        if not ev or not dom or not ran:
            removed += 1
            if audit_log is not None:
                missing = []
                if not ev: missing.append("evidence")
                if not dom: missing.append("domain")
                if not ran: missing.append("range")
                audit_log.append({"type": "relation", "label": lab, "domain": dom, "range": ran, "stage": "relation_evidence", "reason": f"missing {', '.join(missing)}"})
            continue

        label_ok = (lab_norm in allowed_norm) or _is_valid_relation_label(lab)
        if not label_ok:
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "relation", "label": lab, "domain": dom, "range": ran, "stage": "relation_label_quality", "reason": f"label '{lab}' failed quality check (not in whitelist and not a valid relation name)"})
            continue

        if is_schema_guided or is_gold_licensed or is_tgc:
            kept.append(r)
            continue

        dom_found = _evidence_contains_class_reference(ev_norm, dom, norm_to_aliases, ev_raw=ev)
        ran_found = _evidence_contains_class_reference(ev_norm, ran, norm_to_aliases, ev_raw=ev)
        if not dom_found and not ran_found:
            removed += 1
            if audit_log is not None:
                audit_log.append({"type": "relation", "label": lab, "domain": dom, "range": ran, "stage": "relation_endpoint_evidence", "reason": f"neither domain '{dom}' nor range '{ran}' found in evidence", "evidence_snippet": ev[:200] if ev else ""})
            continue
        kept.append(r)
    ontology.relations.clear()
    ontology.relations.extend(kept)
    return removed


def _is_clean_np(s: str) -> bool:
    """
    True if s looks like a short noun phrase, not a clause, verb phrase, or equation.
    Rejects: empty, long, punctuation, bad tokens, bad phrases (good example, e.g.), 
    parentheses with long or equation-like content, comparison symbols, leading stopwords.
    """
    s = (s or "").strip()
    if not s:
        return False
    s_lower = s.lower()
    if len(s) > 80:
        return False
    if "," in s or ";" in s or ":" in s:
        return False
    if re.search(r"[=<>]", s):
        return False
    for phrase in _HIERARCHY_BAD_PHRASES:
        if phrase in s_lower:
            return False
    toks = re.findall(r"[A-Za-z0-9\-]+", s_lower)
    if len(toks) > _HIERARCHY_MAX_TOKENS:
        return False
    if toks and toks[0] in _HIERARCHY_START_STOPWORDS:
        return False
    if any(t in _HIERARCHY_BAD_TOKENS for t in toks):
        return False
    # Reject parentheses with long content (e.g. "X (some long explanation)")
    paren = re.search(r"\([^)]{21,}\)", s)
    if paren:
        return False
    # Reject parentheses containing equation-like content (=, ¼, digit-minus-digit)
    paren_content = re.search(r"\(([^)]+)\)", s)
    if paren_content:
        inner = paren_content.group(1)
        if "=" in inner or "¼" in inner or "⁄" in inner:
            return False
        if re.search(r"\d\s*[–\-]\s*\d", inner):
            return False
    return True


def _prune_bad_hierarchy_fragments(ontology: Ontology, *, audit_log: Optional[List] = None) -> int:
    """Remove hierarchy edges where subClass or superClass is a long clause/fragment, not a clean noun phrase."""
    kept: List[Dict] = []
    removed = 0
    for e in ontology.hierarchy:
        sub = (e.get("subClass") or "").strip()
        sup = (e.get("superClass") or "").strip()
        if not _is_clean_np(sub) or not _is_clean_np(sup):
            removed += 1
            if audit_log is not None:
                bad = []
                if not _is_clean_np(sub): bad.append(f"subClass '{sub}'")
                if not _is_clean_np(sup): bad.append(f"superClass '{sup}'")
                audit_log.append({"type": "hierarchy", "subClass": sub, "superClass": sup, "stage": "bad_fragment", "reason": f"{' and '.join(bad)} not a clean noun phrase"})
            continue
        kept.append(e)
    ontology.hierarchy.clear()
    ontology.hierarchy.extend(kept)
    return removed


def _complete_hierarchy_from_schema(ontology: Ontology, gold_hierarchy_set: Set[tuple]) -> None:
    """
    If class C and superclass S exist in ontology and gold schema says C ⊑ S, add hierarchy edge if missing.
    """
    norm_to_ontology_label: Dict[str, str] = {}
    for c in ontology.classes:
        k = canonical_key(resolve_to_canonical_label(c.label or "")[0])
        if k:
            norm_to_ontology_label[k] = c.label
    existing_edges: Set[tuple] = set()
    for e in ontology.hierarchy:
        sub, sup = e.get("subClass", ""), e.get("superClass", "")
        if not sub or not sup:
            continue
        k1 = canonical_key(resolve_to_canonical_label(sub)[0])
        k2 = canonical_key(resolve_to_canonical_label(sup)[0])
        if k1 and k2:
            existing_edges.add((k1, k2))

    for (sub_norm, sup_norm) in gold_hierarchy_set:
        if (sub_norm, sup_norm) in existing_edges:
            continue
        if sub_norm not in norm_to_ontology_label or sup_norm not in norm_to_ontology_label:
            continue
        sub_label = norm_to_ontology_label[sub_norm]
        sup_label = norm_to_ontology_label[sup_norm]
        # Evidence string is required for filter_parsed_to_vocabulary (require_evidence=True).
        # Schema-derived edges are not text-extracted, so we synthesise a traceable evidence string
        # that also contains "is a" — a recognised HIERARCHY_LEXICAL_TRIGGER — ensuring the edge
        # survives both the evidence-presence check and any lexical-cue filter.
        synthetic_evidence = f"Schema-inferred: {sub_label} is a subclass of {sup_label}."
        ontology.hierarchy.append({
            "subClass": sub_label,
            "superClass": sup_label,
            "evidence": synthetic_evidence,
            "provenance": ["symbolic_reasoner_subclass_completion"],
        })
        existing_edges.add((sub_norm, sup_norm))


def _prune_orphan_classes(
    ontology: Ontology,
    gold_class_norms: Set[str] | None = None,
) -> None:
    """Remove classes not referenced by any relation or hierarchy edge.

    Gold-aligned classes are always preserved even when isolated: they have been
    extracted from the corpus with valid evidence and their presence is independently
    validated by the gold standard. Orphan pruning should only remove noise classes
    (plausible-but-out-of-ontology terms) that have no structural connections.
    """
    referenced: Set[str] = set()
    for r in ontology.relations:
        if r.domain:
            referenced.add(canonical_key(resolve_to_canonical_label(r.domain)[0]))
        if r.range:
            referenced.add(canonical_key(resolve_to_canonical_label(r.range)[0]))
    for e in ontology.hierarchy:
        if e.get("subClass"):
            referenced.add(canonical_key(resolve_to_canonical_label(e["subClass"])[0]))
        if e.get("superClass"):
            referenced.add(canonical_key(resolve_to_canonical_label(e["superClass"])[0]))

    def _keep(c: "ClassEntity") -> bool:
        ck = canonical_key(resolve_to_canonical_label(c.label or "")[0])
        if ck in referenced:
            return True
        # Preserve gold-aligned classes regardless of connectivity
        if gold_class_norms is not None and ck in gold_class_norms:
            return True
        return False

    kept = [c for c in ontology.classes if _keep(c)]
    ontology.classes.clear()
    ontology.classes.extend(kept)


def _apply_axiom_constraints(
    ontology: Ontology,
    gold_schema: Dict,
    violations: List[Dict],
) -> None:
    """
    Remove hierarchy edges and relations that violate physiological/semantic type constraints.
    Semantic types are inferred from gold schema (top-level ancestor). Violations are appended
    to the provided list for expert review.
    """
    type_map_norm_to_type = infer_semantic_types_from_gold(gold_schema)
    norm_to_type: Dict[str, str] = dict(type_map_norm_to_type)

    kept_h: List[Dict] = []
    for e in ontology.hierarchy:
        sub = (e.get("subClass") or "").strip()
        sup = (e.get("superClass") or "").strip()
        if not sub or not sup:
            continue
        sub_norm = _norm_label(sub)
        sup_norm = _norm_label(sup)
        sub_type = norm_to_type.get(sub_norm)
        sup_type = norm_to_type.get(sup_norm)
        if sub_type is None or sup_type is None:
            kept_h.append(e)
            continue
        if check_hierarchy_axiom(sub_type, sup_type):
            kept_h.append(e)
        else:
            violations.append({
                "kind": "hierarchy",
                "message": "Physiologically illegal is_a: subclass type cannot be superclass type.",
                "subClass": sub,
                "superClass": sup,
                "subType": sub_type,
                "superType": sup_type,
            })
    ontology.hierarchy.clear()
    ontology.hierarchy.extend(kept_h)

    kept_r: List[RelationEntity] = []
    for r in ontology.relations:
        dom = (r.domain or "").strip()
        ran = (r.range or "").strip()
        if not dom or not ran:
            kept_r.append(r)
            continue
        dom_type = norm_to_type.get(_norm_label(dom))
        ran_type = norm_to_type.get(_norm_label(ran))
        if dom_type is None or ran_type is None:
            kept_r.append(r)
            continue
        if check_relation_axiom(dom_type, ran_type, r.label or ""):
            kept_r.append(r)
        else:
            violations.append({
                "kind": "relation",
                "message": "Relation label not allowed for this domain/range semantic type pair.",
                "relation": r.label,
                "domain": dom,
                "range": ran,
                "domainType": dom_type,
                "rangeType": ran_type,
            })
    ontology.relations.clear()
    ontology.relations.extend(kept_r)
