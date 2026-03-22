"""Optional scope filter: suppress administrative/organisational content when extracting clinical ontology.

When enabled (e.g. config scope_filter=True), document text is filtered: paragraph-level (drop
admin-heavy paragraphs) then line-level cleanup. Chunk-level filtering uses both admin score and
clinical-positive score so mixed chunks are classified correctly.

Chunk-level: drop if admin high and clinical low; keep if clinical high even if some admin.
Set config min_clinical_score when scope_filter is True to drop low-density chunks.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Default minimum clinical density score when strict chunk filtering is desired (config: min_clinical_score).
# 1.0 is recommended for strict BrainIT clinical runs; 0.0 is permissive.
DEFAULT_MIN_CLINICAL_SCORE_STRICT = 1.0

# Clinical-positive vocabulary (Neuro-ICU / BrainIT). Used in score_chunk_clinical_density.
# Strong terms count 1.0; weak (broad) terms count 0.5.
_CLINICAL_VARIABLE_KEYWORDS_STRONG = frozenset({
    "heart rate", "respiration rate", "blood pressure", "arterial pressure", "mean arterial pressure", "map",
    "intracranial pressure", "icp", "cpp", "cerebral perfusion pressure", "ct scan",
    "gcs", "glasgow", "pupil", "temperature", "brain temperature", "respiration", "oxygen",
    "microdialysis", "sedation", "sedation levels", "ventilation", "haematology", "biochemistry",
    "secondary insult", "hypotension", "hypertension", "intracranial hypertension",
    "therapy", "vasopressors", "blood gases", "minute by minute",
    "monitoring", "physiological monitoring", "clinical monitoring",
    "sao2", "sjo2", "etco2", "tcd", "pbro2", "cvp",
    "fluids", "fluid input", "fluid output", "fluid input and output",
    "nutrition", "nutritional",
    "condition", "clinical condition",
    "antibiotics", "laboratory values",
    # Moss 2013 (Trusting ICU) framework terms — critical for data-quality paper chunks
    "observation", "observations", "timepoint", "timepoints",
    "session", "sessions",
    "sensor", "sensors",
    "data quality", "data quality assessment",
    "possible error", "probable error",
    "noradrenaline", "adrenaline",
    "sepsis",
    "out-of-range", "out of range",
    "reclassif",                          # "reclassified", "reclassification"
    "clinical context",
    "medical disorder",
    "acceptable range",
    # Güiza 2015 — ICP dose/burden terms
    "icp insult", "icp insults", "pressure burden", "transition curve",
    "cerebrovascular autoregulation", "autoregulation",
    # Shaw 2024 — PbtO2 terms
    "pbto2", "brain tissue oxygen", "partial pressure of brain tissue oxygen",
    # Decraene 2023 — surgical intervention terms
    "decompressive craniectomy", "craniectomy", "cranioplasty",
    "therapy intensity level", "til",
    "barbiturate", "barbiturates", "thiopental",
    "mannitol", "osmotics",
    # Stell 2018 — guideline adherence clinical context
    "guideline adherence", "pressure event",
    "eusig",
    # Depreitere 2018 — CPP variability
    "cerebral perfusion pressure variability",
    # Donald 2012 — hypotensive events
    "hypotensive event", "hypotensive events",
    # Georgatzis 2016 — artefact in physiological data
    "artefact", "artefacts", "artefactual",
    # General TBI clinical terms (across all papers)
    "traumatic brain injury", "tbi", "severe tbi",
    "glasgow outcome", "gose", "gos",
    "mortality", "morbidity",
    "neuro-intensive care", "neurointensive care", "neuro-icu",
    "neurosurgery", "neurosurgical",
    "analgesia", "paralysis",
    "volume expansion", "inotropes",
    "hypothermia", "steroids",
    "anti-hypertensives", "anti-pyretics",
})
_CLINICAL_VARIABLE_KEYWORDS_WEAK = frozenset({"core dataset", "outcome"})

# Strong clinical terms (concrete parameters/variables). Count fully (1.0) in clinical score.
# "monitoring" promoted to strong: it is a core BrainIT concept, not a weak/generic term.
_CLINICAL_TERMS_STRONG = (
    "heart rate", "respiration rate", "mean arterial pressure", "icp", "cpp",
    "gcs", "pupil", "ct-scan", "ct scan", "pulse oximetry", "temperature", "cvp",
    "etco2", "sjo2", "tcd", "pbro2", "brain temperature", "microdialysis",
    "sedation", "sedation levels", "vasopressors", "antibiotics", "blood gases", "biochemistry",
    "haematology", "systemic hypotension", "intracranial hypertension",
    "arterial hypotension", "secondary insults",
    "monitoring", "physiological monitoring", "ventilation",
    "nutrition", "nutritional",
    "fluids", "fluid input", "fluid input and output", "fluid output",
    "condition", "clinical condition",
    "laboratory values", "sao2",
    # Moss 2013 (Trusting ICU) framework terms
    "observation", "observations", "timepoint", "timepoints",
    "session", "sessions",
    "sensor", "sensors",
    "data quality", "data quality assessment",
    "possible error", "probable error",
    "noradrenaline", "adrenaline",
    "sepsis",
    "out-of-range", "out of range",
    "reclassif",
    "clinical context",
    "medical disorder",
    "acceptable range",
    # Güiza 2015, Shaw 2024, Decraene 2023, Stell 2018, Depreitere 2018
    "pbto2", "brain tissue oxygen", "autoregulation",
    "decompressive craniectomy", "craniectomy", "cranioplasty",
    "barbiturate", "barbiturates", "thiopental", "mannitol",
    "icp insult", "pressure burden",
    "hypotensive event", "hypotensive events",
    "artefact", "artefacts",
    "traumatic brain injury", "tbi",
    "glasgow outcome", "gose",
    "neuro-intensive care", "neurointensive care",
    "analgesia", "paralysis", "volume expansion", "inotropes",
    "hypothermia", "anti-hypertensives", "anti-pyretics",
    "guideline adherence", "eusig",
)

# Broad terms that can appear in mixed/governance context. Count at 0.5 weight.
_CLINICAL_TERMS_WEAK = ("core dataset", "outcome")
_WEAK_TERM_WEIGHT = 0.5

# Section labels that indicate clinical content (bonus to clinical density score).
_CLINICAL_SECTION_BONUS = frozenset({
    "core dataset", "outcome", "minute by minute", "secondary insult", "secondary insults",
    "intensive care management", "demographic and clinical information",
    # Moss 2013 sections
    "data quality", "data quality assessment", "medical disorders", "treatments",
    "observations", "trusting icu", "probable error", "possible error",
    # Papers 3-10: common clinical sections
    "results", "methods", "patients", "patient population", "patient data",
    "clinical data", "monitoring", "monitoring data", "physiological data",
    "icp management", "therapy", "treatment", "surgical management",
    "decompressive craniectomy", "autoregulation", "cerebral perfusion",
    "pressure burden", "hypotensive events", "artefact detection",
    "clinical characteristics", "baseline characteristics",
})

# Sections that match known dataset sections (BrainIT / Piper 2003). Always keep or strongly prefer.
# These are much closer to the ontology you want.
_CLINICAL_DATASET_SECTIONS = frozenset({
    "demographic and clinical information",
    "minute by minute monitoring information",
    "minute by minute",
    "intensive care management information",
    "intensive care management",
    "secondary insult treatment information",
    "secondary insult treatment",
    "secondary insults",
})

# Sections that indicate governance/Part A content. Chunks with these sections are dropped for strict clinical.
# BrainIT Part A: group structure, centres, database, forms, ethics, validation, publication.
# Extended from deep reading of all 10 BrainIT papers.
_GOVERNANCE_SECTIONS = frozenset({
    "part a",
    "group formation",
    "membership",
    "steering group",
    "technical group",
    "project group",
    "database access",
    "centre membership",
    "center membership",
    "ethics approval",
    "data validation",
    "publication criteria",
    "group coordination",
    # Piper 2003 Part A sub-sections
    "legal status",
    "group funding",
    "group membership",
    "group management",
    "regional coordination",
    "joint authorship guidelines",
    "benefits of membership",
    "benefits of individual membership",
    "benefits of centre membership",
    "benefits of center membership",
    "guidelines to database access and publication",
    "database access=analysis criteria",
    # Common paper end-matter sections (all papers)
    "investigators and participating centres",
    "investigators and participating centers",
    "appendix",
    "appendix a",
    "supplementary materials",
    "statement of ethics",
    "consent for publication",
    "availability of materials",
    "author agreement",
    "declaration of competing interest",
    "competing interests",
    "funding sources",
    # BrainIT governance sections (Piper 2003)
    "project management",
    "collaborative groups",
    "internet registration",
    "data elements definition",
    "data collection software",
    "data collection protocols",
    "data collection methods",
    "data analysis methodologies",
    "data quality control",
    "bedside monitoring types",
    "health care technology",
})

# Phrases that indicate governance-heavy chunk (skip extraction for strict clinical).
# When chunk has 2+ of these and low clinical score, drop.
# Extended from deep reading of all 10 BrainIT papers.
_GOVERNANCE_DOMINANCE_PHRASES = (
    "steering group",
    "technical group",
    "group formation",
    "centre membership",
    "multi-centre",
    "multicentre",
    "sql database",
    "software tool",
    "ethics approval",
    "data validation",
    "publication criteria",
    "group coordination",
    "brainit group",
    "centre membership",
    "center membership",
    "data contributing centres",
    "data contributing centers",
    # Piper 2003 / 2010 governance infrastructure
    "non-profit",
    "not for profit",
    "grant funding",
    "grant funded",
    "legal status",
    "authorship guidelines",
    "joint authorship",
    "operational strategy",
    "data validation nurse",
    "anonymisation",
    "anonymization",
    "patient identifiers",
    "regional coordinator",
    "data contributing",
    "web-services",
    "hand-held pda",
    # Publication infrastructure (Piper 2010, AVERT-IT)
    "grid middleware",
    "grid services",
    "grid technology",
    "avert-it",
    "hospital firewalls",
    "kelvin connect",
    "national escience centre",
    # Semantic web / data technology (Moss 2013)
    "linked data",
    "rdf triples",
    "semantic web",
    "sparql",
    "owl ontology",
    "ontology engineering",
    # Research methodology / study design noise
    "principal investigator",
    "clinical drug trials",
    "helsinki accords",
    "helsinki declaration",
    "medical device companies",
    "multivariate analysis",
    "logistic regression",
    # Statistical methodology (papers 4, 7, 8, 9, 10, 11)
    "mann-whitney",
    "wilcoxon",
    "fisher's exact",
    "shapiro-wilk",
    "chi-square",
    "chi-squared",
    "bonferroni",
    "receiver operating characteristic",
    "kaplan-meier",
    "cross-validation",
    "leave-one-out",
    "monte carlo",
    "sensitivity analysis",
    "marginal effects",
    "cubic regression",
    "multiple imputation",
    # Software tools
    "matlab",
    "ibm spss",
    "smartpls",
    "icm+ software",
    # Study design / ethics
    "inclusion criteria",
    "exclusion criteria",
    "informed consent",
    "waiver of consent",
    "institutional review board",
    "ethics committee approval",
    "consent for publication",
    "author agreement",
    "competing interest",
    "conflict of interest",
    # Publishing / book metadata
    "sciencedirect",
    "journal homepage",
    "springer nature",
    "springer-verlag",
    "springer international",
    "product liability",
    "acid-free paper",
    # Data infrastructure
    "data acquisition",
    "data management system",
    "surveymonkey",
    # Signal processing
    "fourier transform",
    "wavelet analysis",
    # BrainIT governance / organisational concepts
    "brainit group",
    "brainit network",
    "brainit project",
    "project management",
    "collaborative group",
    "internet registration",
    "registration form",
    "non-profit organisation",
    "non-profit organization",
    "data elements definition",
    "data collection software",
    "data collection protocol",
    "data collection method",
    "data collection tool",
    "data analysis methodology",
    "data quality control",
    "computer based sampling",
    "bedside monitoring type",
    "health care technology",
    "sql database",
    "mailbase",
)

# Headings or line-level phrases that indicate out-of-scope (administrative) content.
# When a line is primarily one of these (after normalisation), it can be suppressed.
# Extended from deep reading of all 10 BrainIT papers — covers headings from Piper 2003/2010
# governance sections, publication end-matter across all papers, and IT infrastructure sections.
_ADMIN_HEADINGS = frozenset({
    # BrainIT governance (Piper 2003 Part A)
    "group formation", "membership", "steering group", "technical group", "project group",
    "database access", "publication criteria", "data validation staff", "data validation project",
    "ethics approval", "author contributions", "acknowledgements", "acknowledgments",
    "conflict of interest", "funding", "references", "correspondence", "reprints",
    "individual membership", "centre membership", "center membership", "brainit group",
    "validated data", "unvalidated data", "data quality control", "pharmaceutical industry",
    "database", "data validation", "group coordination structure",
    # Additional Piper 2003 Part A sub-headings
    "legal status", "group funding", "group membership", "group management",
    "regional coordination", "joint authorship guidelines",
    "benefits of individual membership", "benefits of centre membership",
    "benefits of center membership", "benefits of membership",
    "guidelines to database access and publication",
    # Piper 2010 IT/infrastructure sections
    "network structure", "data collection tools", "data validation process",
    "it tool limitations", "lessons learned",
    # Common paper end-matter (appears across all 10 papers)
    "investigators and participating centres", "investigators and participating centers",
    "declaration of competing interest", "competing interests",
    "statement of ethics", "consent for publication", "availability of materials",
    "supplementary materials", "author agreement", "funding sources",
    "appendix", "appendix a", "comments",
    "keywords",
    # Stell 2018 non-clinical framework sections
    "applied technology", "framework technology", "system architecture",
    "clinical result presentation", "related literature",
    # Moss 2013 semantic web / data quality sections
    "linked data", "semantic web", "rdf", "sparql queries", "ontology",
    "data integration", "provenance", "data quality rules",
    "recording configuration", "error annotation", "provenance annotation",
    # Research methodology headings (all 11 papers)
    "statistical analysis", "statistical methods", "study limitations",
    "data analysis", "methodology", "study design", "study population",
    "eligibility", "sample size analysis", "model selection",
    "multivariate analysis", "visualization method",
    "bann construction", "false positive correction", "roc confidence intervals",
    "predictive accuracy", "sensitivity analysis",
    # Common paper sections (all 11 papers)
    "abstract", "introduction", "methods", "materials and methods",
    "results", "discussion", "conclusion", "conclusions",
    "discussion and conclusion",
    # Book-chapter metadata (papers 5, 7, 8 are book chapters)
    "preface", "contents", "author index", "subject index",
    "take-home message", "a r t i c l e i n f o",
    # Data/IT infrastructure sections
    "data source", "data preparation", "data collection",
    "data elements definition", "data collection software",
    "data collection protocols", "data collection methods",
    "data analysis methodologies", "data quality control",
    "computer based sampling", "bedside monitoring types",
    "health care technology", "sql database",
    # Paper-specific methodology sections (papers 4, 7, 9, 10)
    "patient data", "hypotension definition",
    "bann construction and testing", "bann test runs",
    "test set model assessment", "phase i model assessment",
    "false positive suppression", "episode prediction",
    "sample size analysis", "roc confidence intervals",
    "model selection", "predictive accuracy",
    "clinical tests", "data push",
    # Ethics / consent headings (papers 4, 9, 10)
    "ethics approval", "consent for publication",
    "availability of materials", "author contributions",
    "author agreement", "statement of ethics",
    "declaration of competing interest", "competing interests",
    "conflict of interest statement", "affiliations",
    "funding sources",
    # Publishing headings
    "received", "accepted", "published online",
    "handling editor", "corresponding author",
    "supplementary material", "supplementary appendix",
    "electronic supplementary material",
})

# Phrases that if they dominate a line (e.g. line contains little else), suggest suppression.
# Extended from deep reading of all 10 BrainIT papers.
_ADMIN_PHRASES = (
    # BrainIT governance
    "steering group", "technical group", "common database", "pc based data collection",
    "membership guidelines", "group coordination", "publication criteria",
    "brainit group", "individual membership", "centre membership", "data validation staff",
    "data validation project", "validated data", "unvalidated data", "data quality control",
    "pharmaceutical industry", "dv staff",
    # Piper 2003 / 2010 organisational infrastructure
    "non-profit organisation", "not for profit organisation", "legal status",
    "grant funded", "grant funding", "regional coordinator",
    "data collection tools", "web-services", "web-server",
    "anonymisation routine", "patient identifiers", "hand-held pda",
    "operational strategy", "joint authorship", "authorship guidelines",
    "data contributing centres", "data contributing members",
    "data validation nurse", "data validation checking",
    # Piper 2010 Grid/IT infrastructure
    "grid middleware", "grid services", "grid technology",
    "avert-it project", "hospital firewalls", "kelvin connect",
    "national escience centre",
    # Publication metadata (common across all papers)
    "corresponding author", "supplementary material", "electronic supplementary material",
    "on behalf of the brainit group",
    "springer-verlag", "elsevier", "scitepress",
    "open access article",
    "competing financial interest", "personal relationships",
    # Semantic web / data technology (Moss 2013)
    "linked data", "rdf triples", "semantic web", "sparql", "owl ontology",
    "ontology engineering", "data integration platform",
    # Research methodology / study design
    "principal investigator", "post-hoc hypothesis", "post hoc hypothesis",
    "pilot data collection", "medical device companies",
    "clinical drug trials", "helsinki accords", "helsinki declaration",
    "multivariate analysis", "multivariate logistic", "logistic regression",
    "research consortium", "research group",
    # Software / tool / database references (papers 3-11)
    "matlab release", "ibm spss", "smartpls", "program statistica",
    "glmmtmb package", "rocr package", "emmeans package", "mice package",
    "icm+ software",
    # Statistical methodology (papers 4, 7, 8, 9, 10, 11)
    "mann-whitney u test", "wilcoxon signed", "fisher's exact test",
    "shapiro-wilk test", "pearson correlation", "spearman correlation",
    "chi-square test", "chi-squared test", "receiver operating characteristic",
    "area under the curve", "kaplan-meier survival", "bonferroni correction",
    "one-way anova", "cross-validation", "leave-one-out",
    "bayesian artificial neural network", "monte carlo simulation",
    "marginal effects", "estimated marginal means", "sensitivity analysis",
    "multiple imputation", "cubic regression spline",
    "compound symmetry correlation", "multi-level model",
    # Study design (papers 4, 9, 10)
    "inclusion criteria", "exclusion criteria", "informed consent",
    "waiver of consent", "institutional review board",
    "multi-centre research ethics", "ethics committee approval",
    "consent for publication", "author contributions", "author agreement",
    "declaration of competing interest", "conflict of interest statement",
    "statement of ethics", "blinded-endpoint",
    # Publishing / book metadata (papers 5, 7, 8 are book chapters)
    "printed on acid-free paper", "all rights are reserved",
    "contents lists available at sciencedirect", "journal homepage",
    "product liability", "typesetting:", "springer nature",
    "springer international publishing", "springer science",
    # Research projects / consortia
    "center-tbi", "avert-it project", "nemo project",
    "decra trial", "best-trip trial",
    "brain trauma foundation", "cerebral autoregulation network",
    # Data management / acquisition sections
    "data acquisition", "data capture", "data management system",
    "patient data management system", "surveymonkey questionnaire",
    # Signal processing methodology (papers 5, 7)
    "short-time fourier transform", "wavelet analysis",
    "linear interpolation", "resampling",
    # BrainIT governance / organisational concepts (Piper 2003/2010)
    "brainit group", "brainit network", "brainit project",
    "project management", "collaborative group",
    "internet registration form", "registration form",
    "non-profit organisation", "non-profit organization",
    "data elements definition file", "data collection software",
    "data collection protocols", "data collection methods",
    "data collection tools", "data analysis methodologies",
    "data quality control", "computer based sampling",
    "bedside monitoring types", "health care technology",
    "sql database", "multi-centre trial",
)

# Blacklist terms (upstream scope filter): lines/paragraphs containing these are treated as admin.
# Uses compound/specific phrases to avoid destroying legitimate clinical lines that contain
# common words like "group", "trials", or "validation" in a clinical context.
# Extended from deep reading of all 10 BrainIT papers.
_SCOPE_BLACKLIST_TERMS = frozenset({
    # BrainIT governance terms (use compound phrases to avoid clinical false positives)
    "centre membership",
    "center membership",
    "individual membership",
    "membership certificate",
    "membership guidelines",
    "benefits of membership",
    "steering group",
    "ethics committee",
    "ethics approval",
    "consortium administration",
    "data validation staff",
    "data validation nurse",
    "data validation checking",
    "validation workflow",
    "publication criteria",
    # Additional governance/organisational terms (Piper 2003/2010)
    "not for profit",
    "grant funding",
    "authorship guidelines",
    "joint authorship",
    "regional coordinator",
    "operational strategy",
    "data contributing centres",
    "data contributing members",
    # Publication metadata (all papers)
    "corresponding author",
    "supplementary material",
    "competing interest",
    "financial interest",
    "open access article",
    "springer-verlag",
    "elsevier",
    "scitepress",
    # Semantic web / data technology (Moss 2013 noise)
    "linked data",
    "rdf triples",
    "semantic web",
    "sparql query",
    "owl ontology",
    "ontology engineering",
    # Research methodology / study design (all 11 papers)
    "principal investigator",
    "post-hoc hypothesis",
    "post hoc hypothesis",
    "pilot data collection",
    "medical device companies",
    "clinical drug trials",
    "helsinki accords",
    "helsinki declaration",
    "multivariate analysis",
    "multivariate logistic",
    "logistic regression",
    # Software / tool names (papers 3-11)
    "matlab release",
    "ibm spss",
    "smartpls",
    "program statistica",
    "icm+ software",
    "glmmtmb package",
    "rocr package",
    "emmeans package",
    "mice package",
    # Statistical tests / methodology (papers 4, 7, 8, 9, 10, 11)
    "mann-whitney u test",
    "wilcoxon signed",
    "fisher's exact",
    "shapiro-wilk test",
    "pearson correlation coefficient",
    "spearman correlation",
    "chi-square test",
    "chi-squared test",
    "kaplan-meier survival",
    "bonferroni correction",
    "one-way anova",
    "receiver operating characteristic",
    "area under the curve",
    "cross-validation",
    "leave-one-out",
    "sensitivity analysis",
    "monte carlo simulation",
    # Research projects / consortia
    "center-tbi",
    "avert-it project",
    "nemo project",
    "decra trial",
    "best-trip trial",
    "impact score",
    "crash prediction",
    # Book / publishing metadata (papers 5, 7, 8 are book chapters)
    "printed on acid-free paper",
    "all rights are reserved by the publisher",
    "typesetting:",
    "contents lists available at sciencedirect",
    "journal homepage",
    # Study design terms
    "inclusion criteria",
    "exclusion criteria",
    "informed consent",
    "waiver of consent",
    "institutional review board",
    "multi-centre research ethics",
    # BrainIT group / network / project governance (Piper 2003/2010)
    "brainit group",
    "brainit network",
    "brainit project",
    "project management",
    "collaborative group",
    "internet registration form",
    "registration form",
    "non-profit organisation",
    "non-profit organization",
    "data elements definition file",
    "data collection software",
    "data collection protocols",
    "data collection methods",
    "data collection tools",
    "data analysis methodologies",
    "data quality control",
    "data contributing",
    "health care technology",
    "bedside monitoring type",
    "computer based sampling",
    "sampling technique",
    "multi-centre trial",
    "multicenter trial",
    "multicentre trial",
    "sql database",
    "mailbase",
    "web-services",
})


# Regex patterns for publication metadata lines (author affiliations, DOIs, copyright, page headers).
# These appear across all 10 BrainIT papers and are pure noise for ontology extraction.
_PUB_METADATA_PATTERNS = (
    re.compile(r"doi\s*[:.]?\s*10\.\d{4}/", re.IGNORECASE),      # DOI lines (inline or leading)
    re.compile(r"^https?://doi\.org/", re.IGNORECASE),            # DOI URLs
    re.compile(r"^received:?\s+\d|^accepted:?\s+\d|^published\s+online", re.IGNORECASE),
    re.compile(r"^available\s+online\s+\d", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),                                      # bare page numbers
    re.compile(r"^#\s*springer|^\u00a9\s*\d{4}|^copyright\b", re.IGNORECASE),
    re.compile(r"^e-mail\s*(address)?:?\s*\S+@\S+", re.IGNORECASE),
    re.compile(r"isbn:?\s", re.IGNORECASE),                       # ISBN references
    re.compile(r"issn:?\s", re.IGNORECASE),                       # ISSN references
    re.compile(r"^\d{1,3}\.\s+\w.{5,}\(\d{4}\)"),                # numbered reference lines (e.g. "1. Author et al (2004)...")
)


def _normalize_line(s: str) -> str:
    return " ".join((s or "").lower().split())


def _line_is_pub_metadata(line: str) -> bool:
    """True if the line matches a publication metadata pattern (DOI, copyright, affiliations, refs)."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _PUB_METADATA_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _line_contains_blacklist(line: str) -> bool:
    """True if the line contains any blacklist term as a whole word (upstream scope filter)."""
    norm = _normalize_line(line)
    if not norm:
        return False
    for term in _SCOPE_BLACKLIST_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", norm):
            return True
    return False


def _line_has_clinical_terms(norm: str) -> bool:
    """True if the normalised line contains at least one strong clinical keyword."""
    for kw in _CLINICAL_TERMS_STRONG:
        if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", norm):
            return True
    return False


def _line_is_administrative(line: str) -> bool:
    """True if the line is clearly administrative (heading, phrase match, blacklist term, or pub metadata).

    Lines that are exact admin headings are always classified as admin.
    For longer lines (>60 chars), a clinical-rescue check prevents false positives:
    if the line also contains strong clinical terms it is kept as clinical content.
    """
    norm = _normalize_line(line)
    if not norm or len(norm) > 300:
        return False
    if norm in _ADMIN_HEADINGS:
        return True
    has_admin_signal = False
    if _line_contains_blacklist(line):
        has_admin_signal = True
    if not has_admin_signal and _line_is_pub_metadata(line):
        has_admin_signal = True
    if not has_admin_signal:
        for phrase in _ADMIN_PHRASES:
            if phrase in norm and len(norm) < 120:
                has_admin_signal = True
                break
    if not has_admin_signal:
        return False
    if len(norm) > 60 and _line_has_clinical_terms(norm):
        return False
    return True


def _paragraph_admin_ratio(paragraph: str) -> float:
    """Fraction of non-empty lines in the paragraph that are administrative (0.0–1.0)."""
    lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    admin_count = sum(1 for ln in lines if _line_is_administrative(ln))
    return admin_count / len(lines)


def apply_scope_filter(
    text: str,
    *,
    clinical_only: bool = True,
    paragraph_admin_threshold: float = 0.5,
) -> str:
    """
    Remove administrative content so extraction focuses on clinical content.

    Strategy: (1) Split into paragraphs. (2) Drop whole paragraphs if admin-heavy (fraction of
    administrative lines >= paragraph_admin_threshold). (3) Within remaining paragraphs, remove
    administrative lines (line-level cleanup). This reduces prompt contamination from mixed PDF text.
    When clinical_only is False, returns text unchanged.
    """
    if not text or not text.strip():
        return text
    if not clinical_only:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept_paragraphs = []
    for para in paragraphs:
        if _paragraph_admin_ratio(para) >= paragraph_admin_threshold:
            continue
        lines = para.split("\n")
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept_lines.append(line)
                continue
            if _line_is_administrative(stripped):
                continue
            kept_lines.append(line)
        if kept_lines:
            kept_paragraphs.append("\n".join(kept_lines))
    return "\n\n".join(kept_paragraphs)


def apply_scope_filter_to_docs(docs: List[dict], *, clinical_only: bool = True) -> List[dict]:
    """Apply scope filter to each document's 'text' in place. Returns docs."""
    if not clinical_only:
        return docs
    for doc in docs:
        if "text" in doc and doc["text"]:
            doc["text"] = apply_scope_filter(doc["text"], clinical_only=True)
    return docs


def _score_terms(text: str, terms: tuple) -> int:
    """Count how many of the given terms appear in the normalized text (word-boundary matching)."""
    norm = _normalize_line(text)
    return sum(1 for t in terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", norm))


def chunk_scores(chunk: dict) -> tuple[int, float]:
    """
    Return (admin_score, clinical_score) for dual-score routing.
    admin_score: count of admin phrases in chunk text.
    clinical_score: weighted sum—strong terms count 1.0, weak terms (core dataset, monitoring, outcome) count 0.5.
    """
    text = (chunk.get("text") or "").strip()
    norm = _normalize_line(text)
    admin_score = _score_terms(text, _ADMIN_PHRASES)
    strong = sum(1 for t in _CLINICAL_TERMS_STRONG if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", norm))
    weak = sum(1 for t in _CLINICAL_TERMS_WEAK if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", norm))
    clinical_score = strong + _WEAK_TERM_WEIGHT * weak
    return admin_score, clinical_score


def _section_matches_clinical_dataset(chunk: dict) -> bool:
    """True if chunk section matches known clinical dataset sections (always keep)."""
    section = _normalize_line((chunk.get("section") or ""))
    return section in _CLINICAL_DATASET_SECTIONS


def _section_matches_governance(chunk: dict) -> bool:
    """True if chunk section indicates governance/Part A content (drop for strict clinical)."""
    section = _normalize_line((chunk.get("section") or ""))
    return section in _GOVERNANCE_SECTIONS


def _chunk_governance_dominance_score(chunk: dict) -> int:
    """Count of governance-dominance phrases in chunk text (word-boundary). High = skip extraction."""
    text = _normalize_line((chunk.get("text") or ""))
    return sum(1 for p in _GOVERNANCE_DOMINANCE_PHRASES
               if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text))


def _chunk_admin_ratio(chunk: dict) -> float:
    """Fraction of non-empty lines in the chunk that are administrative (0.0–1.0)."""
    text = (chunk.get("text") or "").strip()
    if not text:
        return 1.0
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 1.0
    admin_count = sum(1 for ln in lines if _line_is_administrative(ln))
    return admin_count / len(lines)


def chunk_is_administrative(
    chunk: dict,
    *,
    admin_score_drop_threshold: int = 2,
    clinical_score_keep_threshold: int = 3,
    clinical_score_drop_max: int = 1,
    governance_dominance_drop_threshold: int = 2,
    admin_ratio_threshold: float = 0.5,
    admin_ratio_very_high: float = 0.8,
) -> bool:
    """
    True if the chunk should be dropped for clinical-only extraction.

    Dual-score routing:
    - Drop if section matches governance sections (Part A, group, database, etc.)
    - Drop if governance_dominance_score >= 2 and clinical_score <= 1 (skip governance-heavy chunks)
    - Drop if admin_score >= 2 and clinical_score <= 1 (stricter than before)
    - Keep if clinical_score >= 3 (even if some admin content)
    - Always keep if section matches clinical dataset sections
    - Section in admin headings → drop
    - Fallback: legacy admin_ratio logic for edge cases
    """
    text = (chunk.get("text") or "").strip()
    if not text:
        return True
    section = (chunk.get("section") or "").strip()
    section_norm = _normalize_line(section)
    if section and section_norm in _ADMIN_HEADINGS:
        return True
    # Governance section: drop chunks from Part A, group formation, database, etc.
    if _section_matches_governance(chunk):
        return True
    # Section-aware: always keep chunks from known clinical dataset sections
    if _section_matches_clinical_dataset(chunk):
        return False
    admin_score, clinical_score = chunk_scores(chunk)
    gov_dominance = _chunk_governance_dominance_score(chunk)
    # Keep if clinically rich
    if clinical_score >= clinical_score_keep_threshold:
        return False
    # Drop if governance-dominant and clinically sparse (group, centres, database, software, etc.)
    if gov_dominance >= governance_dominance_drop_threshold and clinical_score <= clinical_score_drop_max:
        return True
    # Drop if admin-heavy and clinically sparse
    if admin_score >= admin_score_drop_threshold and clinical_score <= clinical_score_drop_max:
        return True
    # Fallback: legacy ratio-based logic for mixed chunks
    admin_ratio = _chunk_admin_ratio(chunk)
    if admin_ratio >= admin_ratio_very_high:
        return True
    return admin_ratio >= admin_ratio_threshold


def score_chunk_clinical_density(chunk: dict) -> float:
    """
    Score a chunk by clinical density for strict runs.
    Positive: strong clinical terms (1.0 each), weak terms (0.5 each), section bonus,
    corpus clinical term bonus.
    Negative: administrative line count.
    Chunks with higher score are more clinical; use with min_clinical_score to keep only dense chunks.
    Uses word-boundary matching to avoid double-counting overlapping keywords.
    """
    text = (chunk.get("text") or "").lower()
    section = _normalize_line((chunk.get("section") or ""))
    score = 0.0
    for kw in _CLINICAL_VARIABLE_KEYWORDS_STRONG:
        score += len(re.findall(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text))
    for kw in _CLINICAL_VARIABLE_KEYWORDS_WEAK:
        score += _WEAK_TERM_WEIGHT * len(re.findall(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text))
    if section and section in _CLINICAL_SECTION_BONUS:
        score += 2.0
    # Corpus clinical term boost: terms found in 3+ papers get a 0.3 boost per
    # occurrence. This rewards chunks containing vocabulary known to be clinical
    # across the BrainIT corpus, without requiring a gold standard.
    try:
        from .corpus_vocab import is_corpus_clinical_term, get_all_clinical_terms
        for term, info in get_all_clinical_terms().items():
            if info.get("count", 0) >= 3 and term in text:
                score += 0.3
    except ImportError:
        pass
    lines = [ln.strip() for ln in (chunk.get("text") or "").split("\n") if ln.strip()]
    admin_count = sum(1 for ln in lines if _line_is_administrative(ln))
    score -= admin_count
    return score


# Section priority for reordering (clinical sections first). Lower = higher priority.
_CLINICAL_SECTION_PRIORITY: dict = {
    "demographic and clinical information": 0,
    "minute by minute monitoring information": 1,
    "minute by minute": 2,
    "intensive care management information": 3,
    "intensive care management": 4,
    "secondary insult treatment information": 5,
    "secondary insult treatment": 6,
    "secondary insults": 7,
}


def _chunk_priority_for_reorder(chunk: dict) -> int:
    """Priority for chunk reordering: clinical dataset sections first (0–7), then 999."""
    section = _normalize_line((chunk.get("section") or ""))
    return _CLINICAL_SECTION_PRIORITY.get(section, 999)


def filter_chunks_to_clinical(
    chunks: List[dict],
    *,
    clinical_only: bool = True,
    admin_ratio_threshold: float = 0.5,
    min_clinical_score: Optional[float] = None,
    reorder_clinical_first: bool = True,
) -> List[dict]:
    """
    When clinical_only is True, drop chunks classified as predominantly administrative.
    When min_clinical_score is set, also drop chunks whose clinical density score is below the threshold
    (score from score_chunk_clinical_density: clinical variable mentions + section bonus - admin lines).
    When reorder_clinical_first is True, sort chunks so clinical dataset sections come first.
    Returns the list of chunks to keep; when clinical_only is False, returns chunks unchanged.
    """
    if not clinical_only or not chunks:
        return list(chunks)
    out = [c for c in chunks if not chunk_is_administrative(c, admin_ratio_threshold=admin_ratio_threshold)]
    if min_clinical_score is not None:
        scored = []
        for c in out:
            s = score_chunk_clinical_density(c)
            c_with_score = {**c, "clinical_score": s}
            if s >= min_clinical_score:
                scored.append(c_with_score)
        out = scored
    if reorder_clinical_first and out:
        out = sorted(out, key=_chunk_priority_for_reorder)
    return out
