from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from .assemble import assemble_baseline_prompt
from .assemble_mmr import assemble_mmr_prompt_controlled
from .template_loader import load_template, render_template
from .llm_client import LLMClient
from .parse import parse_output
from .schema import (
    filter_parsed_to_vocabulary,
    filter_hierarchy_to_lexical_cues,
    extract_hierarchy_from_text,
    MIN_EVIDENCE_LENGTH_STRICT,
    HIERARCHY_LEXICAL_TRIGGERS,
)
from .vocabulary import (
    ALLOWED_CLASSES_CORE,
    ALLOWED_RELATIONS_CORE,
    RELATION_ALIASES_CORE,
    ALLOWED_CLASSES_GOVERNANCE,
    ALLOWED_RELATIONS_GOVERNANCE,
    STRICT_RELATIONS_CLINICAL_REFERENCE,
    ALLOWED_CLASSES_PROVENANCE,
    is_governance_input,
    resolve_class_synonyms,
)
from ..ontology.build import _canonical_key
from ..retrieval.embed import build_index
from ..retrieval.similarity import cosine_sim_matrix
from ..retrieval.mmr import mmr_select
from ..retrieval.pool import (
    load_retrieval_pool_concepts,
    load_retrieval_pool_relations,
    load_retrieval_pool_hierarchy,
    load_retrieval_pool_one_shot,
    pool_texts,
    format_pool_example,
    is_provenance_input,
)
from ..ner import extract_suggested_concepts
from ..corpus.clean_chars import strip_control_chars_for_prompt

# Task-pool wiring (see docs/task_pool_wiring.md for design notes):
#   one-shot     → 1 best entry from pool_strict_concepts.json (MMR, k=1)
#   few-shot I   → 3 concept examples from pool_strict_concepts.json (MMR, k=3, gold first)
#   few-shot II  → phase 1: pool_strict_concepts.json; phase 2: pool_strict_relations.json
#   few-shot III → phase 1: concepts; phase 2: relations; phase 3: pool_strict_hierarchy.json


def run_strategy(
    chunks: List[Dict],
    strategy: str,
    llm_client: LLMClient,
    prompt_save_dir: Optional[Path] = None,
    allowed_classes: Optional[List[str]] = None,
    allowed_relations: Optional[List[str]] = None,
    relation_domains: Optional[Dict[str, tuple]] = None,
    gold_ontology: Optional[Dict] = None,
    expansion_threshold: Optional[float] = None,
    *,
    inject_vocab_guardrails: bool = False,
    filter_to_gold_vocabulary: bool = False,
    inject_medical_ner_anchor: bool = False,
    inject_candidate_terms: bool = False,
    strict_relations: bool = False,
    require_label_in_evidence: bool = True,
    clinical_only_routing: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    parsed_chunks: List[Dict] = []
    gold_hierarchy = (gold_ontology or {}).get("hierarchy") or []

    if prompt_save_dir is not None:
        prompt_save_dir.mkdir(parents=True, exist_ok=True)
        (prompt_save_dir / "strategy.txt").write_text(
            f"prompt_strategy={strategy}\nchunks={len(chunks)}\n",
            encoding="utf-8",
        )
        (prompt_save_dir / "README.txt").write_text(
            "Exact prompt text sent to the LLM for this run.\n"
            "strategy.txt = strategy name and chunk count.\n"
            "prompt_chunk_0000.txt, prompt_chunk_0001.txt, ... = full prompt per chunk.\n",
            encoding="utf-8",
        )

    if strategy in ("simple_fewshot", "phased_2step", "one_shot", "phased_3step"):
        concept_pool = load_retrieval_pool_concepts()
        relation_pool = load_retrieval_pool_relations()
        hierarchy_pool = load_retrieval_pool_hierarchy()
        one_shot_pool = load_retrieval_pool_one_shot() if strategy == "one_shot" else []
        corpus_texts = [c["text"] for c in chunks]
        corpus_index = build_index(corpus_texts) if corpus_texts else None
    else:
        concept_pool = []
        relation_pool = []
        hierarchy_pool = []
        one_shot_pool = []
        corpus_texts = []
        corpus_index = None

    def candidate_note(candidates: Optional[List[str]], *, one_shot_style: bool = False) -> str:
        if not candidates:
            return ""
        intro = (
            "Candidate domain terms (contextual hints only — use only if explicitly supported by the text):"
            if one_shot_style
            else "Candidate domain terms (contextual hints from this chunk — use only if supported by the text): "
        )
        sep = "\n" if one_shot_style else ""
        return intro + sep + "; ".join(candidates) + "\n\n"

    def _ner_matches_gold(entity: str, gold_labels_lower: set) -> bool:
        """Check if an NER entity fuzzy-matches any gold vocabulary label."""
        e = entity.lower().strip()
        if not e or len(e) < 2:
            return False
        if e in gold_labels_lower:
            return True
        for gl in gold_labels_lower:
            if e in gl or gl in e:
                return True
        return False

    def filter_ner_to_gold(suggested: List[str], gold_labels: Optional[List[str]]) -> List[str]:
        """When vocab guardrails are active, keep only NER entities that match gold labels."""
        if not gold_labels or not suggested:
            return suggested
        gold_lower = {g.lower().strip() for g in gold_labels}
        return [s for s in suggested if _ner_matches_gold(s, gold_lower)]

    def medical_ner_note(suggested: List[str], max_display: int = 80, *, ner_ran_but_empty: bool = False) -> str:
        """NER-identified clinical concepts — softer wording to avoid conflicting with vocab guardrails."""
        if suggested:
            display = suggested[:max_display]
            return (
                "Suggested concepts (pre-identified by biomedical NER in this text): "
                + ", ".join(display)
                + ".\nThese are entity mentions detected in this chunk. Use them as additional "
                "evidence that a concept is present, but only extract labels from the allowed "
                "class list above.\n\n"
            )
        if ner_ran_but_empty:
            return ""
        return ""

    def build_vocab_and_hints(vocab_prefix: str, ner_text: str, candidate_text: str) -> str:
        """Assemble the vocab + NER + candidate block for insertion AFTER the rules."""
        parts = []
        if vocab_prefix:
            parts.append(vocab_prefix)
        if ner_text:
            parts.append(ner_text)
        if candidate_text:
            parts.append(candidate_text)
        return "".join(parts)

    def merge_parsed(primary: Dict, secondary: Dict) -> Dict:
        """Merge two parsed dicts; dedupe classes by canonical key, relations by (label, domain, range), hierarchy by (subClass, superClass)."""
        merged = {"classes": [], "relations": [], "hierarchy": []}
        seen_class_keys = set()
        seen_relation_keys = set()
        seen_edges = set()
        for item in (primary.get("classes") or []) + (secondary.get("classes") or []):
            label = (item.get("label") or "").strip()
            if not label:
                continue
            key = _canonical_key(label)
            if not key or key in seen_class_keys:
                continue
            seen_class_keys.add(key)
            merged["classes"].append(item)
        for item in (primary.get("relations") or []) + (secondary.get("relations") or []):
            label = (item.get("label") or "").strip()
            dom = (item.get("domain") or "").strip()
            rng = (item.get("range") or "").strip()
            if not label:
                continue
            key = (_canonical_key(label), _canonical_key(dom), _canonical_key(rng))
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            merged["relations"].append(item)
        for item in (primary.get("hierarchy") or []) + (secondary.get("hierarchy") or []):
            sub = (item.get("subClass") or "").strip()
            sup = (item.get("superClass") or "").strip()
            if not sub or not sup:
                continue
            key = (_canonical_key(sub), _canonical_key(sup))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            merged["hierarchy"].append(item)
        return merged

    def _select_from_pool(
        pool: List[Dict],
        chunk_text: str,
        k: int,
        *,
        gold_first: bool = False,
        corpus_texts_ref: Optional[List[str]] = None,
        corpus_index_ref: Optional[object] = None,
    ) -> List[str]:
        """Select examples via MMR from task pool; fallback to corpus if pool empty."""
        if pool:
            pool_text_list = pool_texts(pool)
            index = build_index(pool_text_list)
            k_actual = min(k, len(pool))
            query_vec = index.query([chunk_text])
            sim_to_query = cosine_sim_matrix(query_vec, index.matrix).flatten()
            sim_between = cosine_sim_matrix(index.matrix, index.matrix)
            selected = mmr_select(sim_to_query, sim_between, k=k_actual)
            examples = [format_pool_example(pool[i]) for i in selected if i < len(pool)]
            if gold_first and pool and examples:
                gold_ex = format_pool_example(pool[0])
                examples = [e for e in examples if e != gold_ex]
                examples = [gold_ex] + examples
                examples = examples[:k]
            return examples
        if corpus_index_ref is not None and corpus_texts_ref:
            k_actual = min(k, len(corpus_texts_ref))
            query_vec = corpus_index_ref.query([chunk_text])
            sim_to_query = cosine_sim_matrix(query_vec, corpus_index_ref.matrix).flatten()
            sim_between = cosine_sim_matrix(corpus_index_ref.matrix, corpus_index_ref.matrix)
            selected = mmr_select(sim_to_query, sim_between, k=k_actual)
            return [corpus_texts_ref[i] for i in selected if i < len(corpus_texts_ref)]
        return []

    def build_phase_prompt(
        phase: str,
        chunk_text: str,
        examples: List[str],
        vocab_and_hints: str,
        known_classes: Optional[List[str]] = None,
        expansion_phase: bool = False,
        *,
        hierarchy_in_separate_phase: bool = False,
    ) -> str:
        if phase == "concepts":
            instruction = (
                "### CURRENT TASK: Phase 1 — COMPREHENSIVE CLASS EXTRACTION\n"
                "Extract all clinical and biomedical classes mentioned in the text.\n\n"
                "### CHAIN-OF-THOUGHT (follow these steps before producing output)\n"
                "1. Read each paragraph and identify every clinical noun phrase "
                "(conditions, parameters, therapies, assessments, outcomes, devices, data categories).\n"
                "2. For each candidate, classify it: Is it a clinical condition? A physiological "
                "parameter? A treatment? An assessment? A laboratory value? A nursing intervention? "
                "A data category? A device or sensor?\n"
                "3. Check: does the text provide a verbatim phrase you can cite as evidence?\n"
                "4. If yes, include it. If the concept is purely organisational, governance, "
                "or publication metadata, skip it.\n\n"
                "### CONCEPT TYPES TO LOOK FOR\n"
                "  - Clinical conditions, diseases, injuries, syndromes, secondary insults\n"
                "  - Physiological parameters and monitoring variables (e.g. ICP, MAP, HR, SpO2)\n"
                "  - Treatments, therapies, interventions, drugs, medications\n"
                "  - Surgical procedures and clinical procedures\n"
                "  - Clinical assessments, scores, and scales (e.g. GCS, GOSe, pupil assessment)\n"
                "  - Patient outcomes and outcome measures\n"
                "  - Laboratory values and test categories (e.g. blood gases, biochemistry)\n"
                "  - Nursing interventions and bedside care (e.g. suctioning, sedation)\n"
                "  - Medical devices and sensors\n"
                "  - Data categories and abstract grouping classes (e.g. Monitoring Data, "
                "Therapy, Condition, Outcome, Core Monitoring Parameter, Optional Monitoring Parameter, "
                "Baseline Therapy, Demographic Data, Admission Data, Treatment Data, Outcome Data)\n"
                "  - Clinical guidelines and management protocols\n\n"
                "### LABEL GUIDANCE\n"
                "  - Use short, clean noun-phrase labels.\n"
                "  - Expand common abbreviations: HR → Heart Rate, ICP → Intracranial Pressure, "
                "CPP → Cerebral Perfusion Pressure, GCS → GCS Assessment, GOSe → GOSe Outcome, "
                "PbtO2/PtiO2 → PbrO2, PRx → Pressure Reactivity Index.\n"
                "  - For other abbreviations, expand to their full clinical name.\n\n"
                "### STRUCTURAL COMPLETENESS\n"
                "When the text mentions specific clinical items, also extract their general "
                "category if the text supports it. "
                "Use the text's own categories and groupings. For example, if the text lists "
                "'heart rate, ICP, MAP' under 'core monitoring parameters', extract BOTH the "
                "individual parameters AND 'Core Monitoring Parameter' as a class.\n\n"
                "### DEFINITION REQUIREMENT\n"
                "Every class MUST include a `definition` field: a concise one-sentence scope note "
                "describing what the class represents in the clinical domain.\n"
                "Ground definitions in the text where possible, but you may use standard medical "
                "knowledge to write a clear scope note.\n"
                "Example: {\"label\": \"Intracranial Pressure\", \"definition\": \"A physiological "
                "parameter measuring the pressure inside the cranium, used to monitor patients with "
                "traumatic brain injury.\", \"evidence\": \"...\"}\n\n"
                "### EXTRACTION PRINCIPLE\n"
                "Extract every clinical concept the text explicitly names. "
                "Do not invent concepts not present in the text. "
                "Each concept needs a verbatim evidence quote from the source.\n\n"
                "### SELF-CHECK (before finalising output)\n"
                "- Is every label a clinical TYPE (not an instance or a person's name)?\n"
                "- Are there any duplicate or synonym labels? Merge them.\n"
                "- Does every evidence field contain a verbatim quote from the text?\n\n"
                "Output JSON with keys classes, relations, hierarchy. "
                "Set relations and hierarchy to empty arrays.\n\n"
            )
        elif expansion_phase:
            class_hint = ""
            if known_classes:
                class_hint = (
                    "Allowed classes: " + "; ".join(known_classes) + "\n\n"
                    "IMPORTANT: If you extract a concept that is not in the allowed class list "
                    "but is semantically equivalent to an existing allowed class, map it to the "
                    "closest allowed class. Include an 'original_label' field with the extracted "
                    "term and 'mapping_confidence' (0.0-1.0) indicating semantic similarity.\n\n"
                )
            instruction = (
                "### CURRENT TASK: Phase 2 (Controlled Expansion)\n"
                "Extract additional classes that may be "
                "semantically equivalent to allowed classes. Map them to the closest allowed class. "
                "Output JSON with keys classes, relations, hierarchy.\n\n"
                + class_hint
            )
        elif phase == "hierarchy":
            class_hint = ""
            if known_classes:
                class_hint = (
                    "Known classes (use only these for subclass/superclass): "
                    + "; ".join(known_classes)
                    + "\n\n"
                )
            instruction = (
                "### CURRENT TASK: Phase 3 — HIERARCHY EXTRACTION\n"
                "Extract subclass/superclass (is-a) edges supported by the text. A hierarchy edge "
                "means one class is a TYPE, SUBTYPE, CATEGORY MEMBER, or SPECIFIC INSTANCE of another. "
                "Extract every edge the text explicitly supports with evidence.\n\n"
                "### HIERARCHY DETECTION METHOD\n"
                "Look for ALL of these patterns in the text:\n"
                "  1. Explicit cues: 'such as', 'is a', 'type of', 'kind of', 'form of'\n"
                "  2. Enumeration: 'include:', 'includes:', 'consists of', 'comprises'\n"
                "  3. Category grouping: 'categorized as', 'classified as', 'grouped into'\n"
                "  4. List patterns: items listed under a heading or category name\n"
                "  5. Implicit grouping: when several specific items are mentioned alongside "
                "a general category (e.g., 'therapy: ventilation, sedation' implies both are "
                "subclasses of Therapy; 'core parameters: HR, ICP, MAP' implies all three are "
                "subclasses of Core Monitoring Parameter)\n"
                "  6. Definitional: 'X is a Y', 'X, a type of Y', 'X (a form of Y)'\n"
                "  7. Context-based: if the text organises items into sections or tables, "
                "each section heading implies a parent class for the items listed under it\n"
                "  8. Medical knowledge: TBI is a type of Brain Injury, Hypotension is a "
                "type of Secondary Insult — extract these when the text discusses them together\n\n"
                "For enumeration patterns (e.g. 'include: X, Y, Z'), extract EACH listed "
                "item as a subclass of the parent category.\n"
                "IMPORTANT: If X 'includes' Y, model it as: Y subClassOf X (a hierarchy edge). "
                "Do NOT output 'includes' as a relation label — it belongs in the hierarchy.\n\n"
                "### CHAIN-OF-THOUGHT\n"
                "For each pair of known classes, reason:\n"
                "1. Does the text describe one class as a specialization of another?\n"
                "2. Does the text list one class under another as a member/subtype?\n"
                "3. Even if not explicitly stated, does medical context clearly imply "
                "a parent-child relationship?\n"
                "4. What text evidence supports this parent-child link?\n\n"
                "Be thorough: extract every parent-child relationship the text supports. "
                "Aim for at least one hierarchy edge per class if possible.\n"
                "If the text lists items under a category heading, EACH item is a subClassOf "
                "that category. For example, if 'core monitoring' is followed by 'heart rate, "
                "ICP, MAP', then each is a subClassOf Core Monitoring Parameter.\n"
                "Use ONLY the known classes below for subClass and superClass labels.\n\n"
                "### SELF-CHECK (before finalising output)\n"
                "- Are there any circular edges (A subClassOf B AND B subClassOf A)? Remove them.\n"
                "- Are subClass and superClass labels from the known-classes list?\n"
                "- Does every evidence field contain a verbatim quote from the text?\n\n"
                "Output JSON with keys classes, relations, hierarchy. "
                "Set classes and relations to empty arrays.\n\n" + class_hint
            )
        else:
            class_hint = ""
            if known_classes:
                class_hint = "Known classes (use only these labels): " + "; ".join(known_classes) + "\n\n"
            _relation_block = (
                "### WHAT IS AN ONTOLOGY RELATION?\n"
                "An ontology relation (object property) defines a typed semantic link between two "
                "class types. The DOMAIN is the subject class; the RANGE is the object class. "
                "Think: 'Every instance of [domain] can be linked via [relation] to an instance "
                "of [range].' Relations model the SCHEMA of the domain, not individual facts.\n"
                "Example: 'has monitoring data(Patient → Monitoring Data)' means every Patient "
                "can have Monitoring Data associated with them.\n\n"
                "### RELATION EXTRACTION METHOD (Chain-of-Thought)\n"
                "For each pair of known classes, reason step by step:\n"
                "1. Do these two classes co-occur or interact in the text?\n"
                "2. What is the SEMANTIC NATURE of their link? (measurement, treatment, "
                "assessment, outcome, composition, causation, etc.)\n"
                "3. Which class is the subject (domain) and which is the object (range)?\n"
                "4. Assign a descriptive relation label from the patterns below, or create a "
                "meaningful label if none fits.\n"
                "5. Find an exact text quote that supports this relationship.\n\n"
                "### ONTOLOGY DESIGN PATTERNS (use when the text supports them)\n"
                "These are reusable structural patterns showing how clinical concepts relate.\n"
                "Match them against the text — extract every supported edge.\n\n"
                "**Pattern 1 — Clinical Monitoring:**\n"
                "  Patient --[has monitoring data]--> Monitoring Data\n"
                "  Monitoring Data --[includes]--> Parameter\n"
                "  Observation --[measures parameter]--> Parameter\n"
                "  Observation --[produced by sensor]--> Sensor\n"
                "  Observation --[has quality assessment]--> Data Quality Assessment\n"
                "  Session --[has timepoint]--> Timepoint\n"
                "  Timepoint --[has observation]--> Observation\n\n"
                "**Pattern 2 — Clinical Intervention:**\n"
                "  Patient --[receives therapy]--> Therapy\n"
                "  Therapy --[targets condition]--> Condition\n"
                "  Condition --[triggers intervention]--> Therapy\n"
                "  Patient --[has surgical procedure]--> Surgical Procedure\n"
                "  Patient --[has nursing intervention]--> Nursing Intervention\n\n"
                "**Pattern 3 — Clinical Assessment and Outcome:**\n"
                "  Patient --[has clinical assessment]--> Clinical Assessment\n"
                "  Patient --[has outcome]--> Outcome\n"
                "  Patient --[has laboratory value]--> Laboratory Values\n"
                "  Monitoring Data --[monitoring indicates condition]--> Condition\n\n"
                "**Pattern 4 — Patient Context:**\n"
                "  Patient --[has session]--> Session\n"
                "  Patient --[has demographic data]--> Demographic Data\n"
                "  Patient --[has injury mechanism]--> Injury Mechanism\n\n"
                "**Pattern 5 — Data Quality:**\n"
                "  Data Quality Assessment --[affected by treatment]--> Therapy\n"
                "  Data Quality Assessment --[associated with condition]--> Condition\n\n"
                "**Pattern 6 — Composition (model as HIERARCHY, not relation):**\n"
                "  If X includes/comprises Y → extract as hierarchy: Y subClassOf X\n"
                "  Derived Parameter --[derived from]--> Source Parameter\n\n"
                "You SHOULD also extract other relations the text supports beyond these patterns.\n"
                "Do not invent relations not stated or clearly implied in the text.\n\n"
                "### CRITICAL: DO NOT USE 'includes' AS A RELATION LABEL\n"
                "If X includes Y, or Y is a type/kind/member of X, model it as HIERARCHY "
                "(Y subClassOf X), NOT as a relation. For example:\n"
                "  - 'Monitoring Data includes Heart Rate' → Heart Rate subClassOf Monitoring Data (hierarchy)\n"
                "  - 'Third-tier Treatment includes Barbiturates' → Barbiturates subClassOf Third-tier Treatment (hierarchy)\n"
                "The labels 'includes', 'consists of', 'comprises', 'is a type of' are NOT valid relation labels.\n\n"
                "### DEFINITION REQUIREMENT\n"
                "Every relation MUST include a `definition` field: a concise one-sentence description "
                "of what this relation means between its domain and range classes.\n"
                "Example: {\"label\": \"has monitoring data\", \"domain\": \"Patient\", \"range\": \"Monitoring Data\", "
                "\"definition\": \"Links a patient to the set of continuously recorded physiological "
                "measurements captured during their ICU stay.\", \"evidence\": \"...\"}\n\n"
                "### RELATION EVIDENCE RULES\n"
                "- Each relation MUST have an 'evidence' field with an exact verbatim quote "
                "from the source text.\n"
                "- The evidence should mention or clearly imply at least one of the two "
                "endpoint classes (domain or range).\n"
                "- Use the ABSTRACT superclass as the range, not specific instances "
                "(e.g., 'Monitoring Data' not 'ICP').\n"
                "- Do NOT reuse the same evidence quote for multiple relations.\n\n"
                "### SELF-CHECK (before finalising output)\n"
                "- Do all domain and range labels match known classes from Phase 1?\n"
                "- Are there any duplicate relations (same domain-label-range)? Remove duplicates.\n"
                "- Does every evidence field contain a verbatim quote from the text?\n\n"
            )
            if hierarchy_in_separate_phase:
                instruction = (
                    "### CURRENT TASK: Phase 2 — RELATION & HIERARCHY EXTRACTION (+ supplementary classes)\n"
                    "Primary task: extract semantic RELATIONS between the known classes below. "
                    "Extract every relation the text explicitly supports with evidence.\n"
                    "Secondary task: if you notice clinical concepts in the text that Phase 1 missed "
                    "(e.g. a concept needed as domain/range), add them to the classes array with evidence.\n"
                    "Also extract hierarchy (subClassOf) edges wherever the text shows parent-child "
                    "relationships — look for cues like 'such as', 'includes', 'type of', 'consists of', "
                    "'categorized as', enumeration lists, or even implied groupings.\n\n"
                    + _relation_block +
                    "Output JSON with keys classes, relations, hierarchy.\n\n"
                    + class_hint
                )
            else:
                instruction = (
                    "### CURRENT TASK: Phase 2 — RELATIONS & HIERARCHY (+ supplementary classes)\n"
                    "Primary task: extract semantic RELATIONS and hierarchy edges between the known classes.\n"
                    "Secondary task: if you notice clinical concepts the text mentions that Phase 1 missed, "
                    "add them to the classes array with evidence.\n\n"
                    + _relation_block +
                    "Output JSON with keys classes, relations, hierarchy.\n\n"
                    + class_hint
                )
        return assemble_mmr_prompt_controlled(
            chunk_text,
            examples,
            vocab_and_hints=vocab_and_hints,
            phase_instruction=instruction,
        )

    for idx, chunk in enumerate(chunks):
        if progress_callback is not None:
            progress_callback(idx, len(chunks), f"Processing chunk {idx + 1} of {len(chunks)}")
        suggested_concepts: List[str] = []
        use_provenance = False
        use_governance = False
        use_core = False
        effective_allowed_classes = allowed_classes
        effective_allowed_relations = allowed_relations

        # Skip reference/bibliography chunks entirely — no useful ontology content
        if chunk.get("section_type") == "skip":
            continue

        if strategy in ("simple_fewshot", "phased_2step", "one_shot", "phased_3step", "baseline"):
            chunk_text = strip_control_chars_for_prompt(chunk.get("text") or "")
            # Section-aware context: tell the LLM what type of section this chunk is from
            section_type = chunk.get("section_type", "unknown")
            section_heading = chunk.get("section", "")
            section_context = ""
            if section_type == "high":
                section_context = (
                    f"[SECTION CONTEXT: This text is from a clinical-definition section"
                    + (f" ('{section_heading}')" if section_heading else "")
                    + ". It is likely rich in domain concepts, parameters, and clinical relationships. "
                    "Extract thoroughly.]\n\n"
                )
            elif section_type == "medium":
                section_context = (
                    f"[SECTION CONTEXT: This text is from a results/discussion section"
                    + (f" ('{section_heading}')" if section_heading else "")
                    + ". Focus on clinical concepts mentioned; ignore statistical values and p-values.]\n\n"
                )
            elif section_type == "low":
                section_context = (
                    f"[SECTION CONTEXT: This text is from a methodology/study-design section"
                    + (f" ('{section_heading}')" if section_heading else "")
                    + ". Extract only clinical concepts (conditions, parameters, therapies). "
                    "Skip study methodology terms (cohort, regression, sample size, etc.).]\n\n"
                )
            suggested_concepts = extract_suggested_concepts(chunk_text) if inject_medical_ner_anchor else []
            # Prepend section context hint for the LLM (section-aware weighting)
            if section_context:
                chunk_text = section_context + chunk_text
            use_provenance = (
                not clinical_only_routing
                and is_provenance_input(chunk_text)
            )
            use_governance = (
                not clinical_only_routing
                and not use_provenance
                and is_governance_input(chunk_text)
            )
            use_core = not use_provenance and not use_governance

            if effective_allowed_classes is None or effective_allowed_relations is None:
                if use_provenance:
                    effective_allowed_classes = list(ALLOWED_CLASSES_PROVENANCE)
                    effective_allowed_relations = []
                elif use_governance:
                    effective_allowed_classes = list(ALLOWED_CLASSES_GOVERNANCE)
                    effective_allowed_relations = list(ALLOWED_RELATIONS_GOVERNANCE)
                else:
                    effective_allowed_classes = list(ALLOWED_CLASSES_CORE)
                    effective_allowed_relations = (
                        list(STRICT_RELATIONS_CLINICAL_REFERENCE)
                        if strict_relations
                        else list(ALLOWED_RELATIONS_CORE)
                    )

            examples: List[str] = []
            examples_concepts: List[str] = []
            examples_relations: List[str] = []
            examples_hierarchy: List[str] = []
            _corpus_ref = corpus_texts if corpus_index is not None and corpus_texts else None
            _index_ref = corpus_index

            if strategy == "one_shot":
                effective_one_shot_pool = one_shot_pool if one_shot_pool else concept_pool
                examples = _select_from_pool(
                    effective_one_shot_pool, chunk_text, 1,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                if not examples and effective_one_shot_pool:
                    examples = [format_pool_example(effective_one_shot_pool[0])]
                if not examples:
                    examples = [
                        "(No example. Ensure resources/pool_one_shot_comprehensive.json exists and has entries.)"
                    ]
                examples_hierarchy = _select_from_pool(
                    hierarchy_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                if not examples_hierarchy and hierarchy_pool:
                    examples_hierarchy = [format_pool_example(p) for p in hierarchy_pool[:3]]
            elif strategy == "simple_fewshot":
                examples = _select_from_pool(
                    concept_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                if len(examples) < 3 and concept_pool:
                    for p in concept_pool:
                        if len(examples) >= 3:
                            break
                        cand = format_pool_example(p)
                        if cand not in examples:
                            examples.append(cand)
                if len(examples) < 3 and corpus_texts:
                    for ct in corpus_texts:
                        if len(examples) >= 3:
                            break
                        if ct not in examples:
                            examples.append(ct)
                examples = examples[:3]
                if not examples and concept_pool:
                    examples = [format_pool_example(p) for p in concept_pool[:3]]
                if not examples:
                    examples = [
                        "(No retrieved examples. Ensure resources/pool_strict_concepts.json exists.)"
                    ]
            elif strategy == "phased_2step":
                examples_concepts = _select_from_pool(
                    concept_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                examples_relations = _select_from_pool(
                    relation_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                examples_hierarchy = _select_from_pool(
                    hierarchy_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                for ex_list, pool_ref in [
                    (examples_concepts, concept_pool),
                    (examples_relations, relation_pool),
                    (examples_hierarchy, hierarchy_pool),
                ]:
                    if len(ex_list) < 3 and pool_ref:
                        for p in pool_ref:
                            if len(ex_list) >= 3:
                                break
                            cand = format_pool_example(p)
                            if cand not in ex_list:
                                ex_list.append(cand)
                    if not ex_list and pool_ref:
                        ex_list[:] = [format_pool_example(p) for p in pool_ref[:3]]
                if not examples_concepts:
                    examples_concepts = [
                        "(No concept examples. Ensure resources/pool_strict_concepts.json exists.)"
                    ]
                if not examples_relations:
                    examples_relations = [
                        "(No relation examples. Ensure resources/pool_strict_relations.json exists.)"
                    ]
            elif strategy == "phased_3step":
                examples_concepts = _select_from_pool(
                    concept_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                examples_relations = _select_from_pool(
                    relation_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                examples_hierarchy = _select_from_pool(
                    hierarchy_pool, chunk_text, 3, gold_first=True,
                    corpus_texts_ref=_corpus_ref, corpus_index_ref=_index_ref,
                )
                for ex_list, pool_ref in [
                    (examples_concepts, concept_pool),
                    (examples_relations, relation_pool),
                    (examples_hierarchy, hierarchy_pool),
                ]:
                    if len(ex_list) < 3 and pool_ref:
                        for p in pool_ref:
                            if len(ex_list) >= 3:
                                break
                            cand = format_pool_example(p)
                            if cand not in ex_list:
                                ex_list.append(cand)
                    if not ex_list and pool_ref:
                        ex_list[:] = [format_pool_example(p) for p in pool_ref[:3]]
                if not examples_concepts:
                    examples_concepts = [
                        "(No concept examples. Ensure resources/pool_strict_concepts.json exists.)"
                    ]
                if not examples_relations:
                    examples_relations = [
                        "(No relation examples. Ensure resources/pool_strict_relations.json exists.)"
                    ]
                if not examples_hierarchy:
                    examples_hierarchy = [
                        "(No hierarchy examples. Ensure resources/pool_strict_hierarchy.json exists.)"
                    ]

            if strategy == "baseline":
                prompt = assemble_baseline_prompt(chunk_text)
            elif strategy == "one_shot":
                template = load_template("one_shot")
                prompt = render_template(template, text=chunk_text, examples="\n\n".join(examples))
            elif strategy == "simple_fewshot":
                prompt = assemble_mmr_prompt_controlled(chunk_text, examples)
            elif strategy == "phased_2step":
                prompt = ""
            elif strategy == "phased_3step":
                prompt = ""
            else:
                chunk_text = strip_control_chars_for_prompt(chunk.get("text") or "")
                suggested_concepts = extract_suggested_concepts(chunk_text) if inject_medical_ner_anchor else []
                prompt = assemble_baseline_prompt(chunk_text)

        # --- Build the vocab + NER + candidate hints block (inserted AFTER rules in template) ---
        vocab_prefix = ""
        classes_for_vocab = effective_allowed_classes or allowed_classes
        relations_for_vocab = effective_allowed_relations if effective_allowed_relations is not None else allowed_relations
        if inject_vocab_guardrails and classes_for_vocab and (relations_for_vocab is not None):
            cls_set = sorted(set(classes_for_vocab))
            from .vocabulary import EXTRACTION_RELATION_LABELS
            rel_set = sorted(set(EXTRACTION_RELATION_LABELS))
            if cls_set:
                rel_lines = []
                rel_lines_typed_only = []
                if relation_domains:
                    for rel in rel_set:
                        domain, range_ = relation_domains.get(rel, (None, None))
                        if domain and range_:
                            typed = f"{rel}({domain}->{range_})"
                            rel_lines.append(typed)
                            rel_lines_typed_only.append(typed)
                        else:
                            rel_lines.append(rel)
                else:
                    rel_lines = list(rel_set)
                if strategy == "one_shot":
                    rels_for_one_shot = rel_lines_typed_only if rel_lines_typed_only else rel_lines
                    vocab_prefix = (
                        "Use ONLY the following class labels:\n"
                        + "; ".join(cls_set)
                        + "\n\nUse ONLY the following relation labels with the stated domain and range:\n"
                        + "; ".join(rels_for_one_shot)
                        + "\n\nIf an item is not supported by the text and does not match the allowed labels above, omit it.\n\n"
                        "Leaf-first rule (important):\n"
                        "Prefer the most specific allowed class labels explicitly supported by the text.\n"
                        "Do not stop at broad container classes such as Monitoring Data, Therapy, or Condition if the text names more specific allowed concepts.\n"
                        "Broad classes may be included only when directly supported by the text or when needed as valid endpoints for a clearly supported relation.\n"
                        "If multiple distinct allowed leaf concepts are explicitly named, extract each of them separately with its own evidence.\n"
                        "Do not treat research methods, feasibility workflow, organisational process, or guideline-adherence calculations as Therapy, Monitoring Data, or other clinical ontology classes.\n\n"
                    )
                else:
                    vocab_prefix = (
                        "Use ONLY the following class labels:\n"
                        + "; ".join(cls_set)
                        + "\nRelations (use exact labels + domain/range):\n"
                        + "; ".join(rel_lines)
                        + "\nIf an item is not in the list, omit it.\n\n"
                        "LEAF-FIRST RULE (important):\n"
                        "- Prefer the most specific class labels mentioned in the text (e.g., ICP/CPP/MAP, Ventilation/Sedation/Fluids/Nutrition).\n"
                        "- Avoid outputting only broad container classes (e.g., Monitoring Data, Therapy, Condition) if the text names specific items.\n"
                        "- You may include container classes when needed as endpoints for allowed relations, but do not stop at containers when leaf items are present.\n"
                        "- If the text clearly mentions multiple distinct allowed leaf items, extract several of them (each with evidence) rather than stopping early.\n"
                        "- Do NOT treat research methods/algorithms/guideline-adherence calculations as Therapy or Monitoring Data.\n\n"
                    )

        candidates_for_prompt = (chunk.get("candidates") or []) if inject_candidate_terms else []
        filtered_ner = (
            filter_ner_to_gold(suggested_concepts, classes_for_vocab)
            if inject_vocab_guardrails and classes_for_vocab
            else suggested_concepts
        )
        ner_text = (
            medical_ner_note(filtered_ner, ner_ran_but_empty=inject_medical_ner_anchor and not filtered_ner)
            if filtered_ner or inject_medical_ner_anchor
            else ""
        )
        candidate_text = candidate_note(candidates_for_prompt, one_shot_style=(strategy == "one_shot"))
        vocab_and_hints = build_vocab_and_hints(vocab_prefix, ner_text, candidate_text)

        # For non-phased strategies, inject vocab+hints into the prompt
        if strategy not in ("phased_2step", "phased_3step") and prompt:
            if "{{VOCAB_AND_HINTS}}" in prompt:
                prompt = prompt.replace("{{VOCAB_AND_HINTS}}", vocab_and_hints.strip())
            else:
                prompt = vocab_and_hints + prompt

        if strategy == "phased_2step":
            prompt_phase1 = build_phase_prompt(
                "concepts",
                chunk_text,
                examples_concepts,
                vocab_and_hints,
            )
            raw_phase1 = llm_client.generate(prompt_phase1)
            parsed_phase1 = resolve_class_synonyms(parse_output(raw_phase1))
            if allowed_classes:
                parsed_phase1_filtered = filter_parsed_to_vocabulary(
                    parsed_phase1,
                    allowed_classes,
                    allowed_relations or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    gold_hierarchy=gold_hierarchy,
                    require_label_in_evidence=require_label_in_evidence,
                )
            else:
                parsed_phase1_filtered = parsed_phase1
            class_labels = [c.get("label") for c in parsed_phase1_filtered.get("classes", []) if c.get("label")]
            prompt_phase2 = build_phase_prompt(
                "relations",
                chunk_text,
                examples_relations,
                vocab_and_hints,
                known_classes=class_labels,
            )
            raw_phase2 = llm_client.generate(prompt_phase2)
            parsed_phase2 = resolve_class_synonyms(parse_output(raw_phase2))
            if allowed_classes:
                parsed_phase2 = filter_parsed_to_vocabulary(
                    parsed_phase2,
                    allowed_classes,
                    allowed_relations or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    gold_hierarchy=gold_hierarchy,
                    require_label_in_evidence=require_label_in_evidence,
                )
            parsed = merge_parsed(parsed_phase1_filtered, parsed_phase2)
            if prompt_save_dir is not None:
                (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase1.txt").write_text(
                    prompt_phase1, encoding="utf-8"
                )
                (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase2.txt").write_text(
                    prompt_phase2, encoding="utf-8"
                )
        elif strategy == "phased_3step":
            prompt_phase1 = build_phase_prompt(
                "concepts",
                chunk_text,
                examples_concepts,
                vocab_and_hints,
            )
            raw_phase1 = llm_client.generate(prompt_phase1)
            parsed_phase1 = resolve_class_synonyms(parse_output(raw_phase1))
            if allowed_classes:
                parsed_phase1_filtered = filter_parsed_to_vocabulary(
                    parsed_phase1,
                    allowed_classes,
                    allowed_relations or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    gold_hierarchy=gold_hierarchy,
                    require_label_in_evidence=require_label_in_evidence,
                )
            else:
                parsed_phase1_filtered = parsed_phase1
            class_labels = [c.get("label") for c in parsed_phase1_filtered.get("classes", []) if c.get("label")]
            prompt_phase2 = build_phase_prompt(
                "relations",
                chunk_text,
                examples_relations,
                vocab_and_hints,
                known_classes=class_labels,
                hierarchy_in_separate_phase=True,
            )
            raw_phase2 = llm_client.generate(prompt_phase2)
            parsed_phase2 = resolve_class_synonyms(parse_output(raw_phase2))
            if allowed_classes:
                parsed_phase2 = filter_parsed_to_vocabulary(
                    parsed_phase2,
                    allowed_classes,
                    allowed_relations or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    gold_hierarchy=gold_hierarchy,
                    require_label_in_evidence=require_label_in_evidence,
                )
            parsed = merge_parsed(parsed_phase1_filtered, parsed_phase2)
            all_class_labels = [c.get("label") for c in parsed.get("classes", []) if c.get("label")]
            # Run Phase 3 whenever at least 2 classes are extracted — the LLM
            # can determine if hierarchy relationships exist. The old trigger-based
            # gating skipped many chunks with valid implicit hierarchy patterns.
            if (
                len(all_class_labels) >= 2
                and examples_hierarchy
            ):
                phase3_known = list(all_class_labels)
                # Only inject gold vocab / forced labels when guardrails are active
                if inject_vocab_guardrails:
                    if allowed_classes and len(phase3_known) < 5:
                        for gc in allowed_classes:
                            if gc not in phase3_known:
                                phase3_known.append(gc)
                    if "Therapy" not in phase3_known and use_core:
                        phase3_known = ["Therapy"] + phase3_known
                    if "Baseline Therapy" not in phase3_known and use_core:
                        phase3_known = ["Baseline Therapy"] + phase3_known
                prompt_phase3 = build_phase_prompt(
                    "hierarchy",
                    chunk_text,
                    examples_hierarchy,
                    vocab_and_hints,
                    known_classes=phase3_known,
                )
                raw_phase3 = llm_client.generate(prompt_phase3)
                parsed_phase3 = resolve_class_synonyms(parse_output(raw_phase3))
                if parsed_phase3.get("hierarchy"):
                    # Accept hierarchy edges that have non-empty evidence
                    # (the chunk was already gated for hierarchy triggers)
                    hierarchy_valid = [
                        e for e in parsed_phase3.get("hierarchy", [])
                        if (e.get("evidence") or "").strip()
                    ]
                    if hierarchy_valid:
                        parsed = merge_parsed(parsed, {"classes": [], "relations": [], "hierarchy": hierarchy_valid})
                if prompt_save_dir is not None:
                    (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase3_hierarchy.txt").write_text(
                        prompt_phase3, encoding="utf-8"
                    )
            if prompt_save_dir is not None:
                (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase1.txt").write_text(
                    prompt_phase1, encoding="utf-8"
                )
                (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase2.txt").write_text(
                    prompt_phase2, encoding="utf-8"
                )
        else:
            # For baseline, one_shot, simple_fewshot, phased_2step: prompt was already built above; do not overwrite.
            if prompt_save_dir is not None:
                (prompt_save_dir / f"prompt_chunk_{idx:04d}.txt").write_text(prompt, encoding="utf-8")
            raw = llm_client.generate(prompt)
            parsed = resolve_class_synonyms(parse_output(raw))
            # Save raw output when parsing returned nothing (for debugging control-char or other parse failures)
            if prompt_save_dir is not None and raw and raw.strip():
                has_any = parsed.get("classes") or parsed.get("relations") or parsed.get("hierarchy")
                if not has_any:
                    (prompt_save_dir / f"raw_failed_chunk_{idx:04d}.txt").write_text(raw, encoding="utf-8", errors="replace")

        if filter_to_gold_vocabulary:
            if (effective_allowed_classes and (effective_allowed_relations is not None)) or (allowed_classes and allowed_relations):
                parsed = filter_parsed_to_vocabulary(
                    parsed,
                    effective_allowed_classes or allowed_classes or [],
                    (effective_allowed_relations if effective_allowed_relations is not None else allowed_relations) or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    gold_hierarchy=gold_hierarchy,
                    require_hierarchy_lexical_cues=False,
                    require_label_in_evidence=require_label_in_evidence,
                )
            elif use_core:
                parsed = filter_parsed_to_vocabulary(
                    parsed,
                    ALLOWED_CLASSES_CORE,
                    ALLOWED_RELATIONS_CORE,
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_aliases=RELATION_ALIASES_CORE,
                    gold_hierarchy=gold_hierarchy,
                    require_hierarchy_lexical_cues=False,
                    require_label_in_evidence=require_label_in_evidence,
                )
            elif use_governance:
                parsed = filter_parsed_to_vocabulary(
                    parsed,
                    ALLOWED_CLASSES_GOVERNANCE,
                    ALLOWED_RELATIONS_GOVERNANCE,
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    gold_hierarchy=gold_hierarchy,
                    require_hierarchy_lexical_cues=False,
                    require_label_in_evidence=require_label_in_evidence,
                )

        # Optional hierarchy-only phase for few-shot strategies: explicitly ask for subclass/superclass.
        filter_classes = effective_allowed_classes or allowed_classes
        filter_relations = effective_allowed_relations if effective_allowed_relations is not None else allowed_relations
        if inject_vocab_guardrails or filter_to_gold_vocabulary:
            if not filter_classes and use_core:
                filter_classes = list(ALLOWED_CLASSES_CORE)
                filter_relations = list(STRICT_RELATIONS_CLINICAL_REFERENCE if strict_relations else ALLOWED_RELATIONS_CORE)
            if not filter_classes and use_governance:
                filter_classes = list(ALLOWED_CLASSES_GOVERNANCE)
                filter_relations = list(ALLOWED_RELATIONS_GOVERNANCE)
        if strategy in ("phased_2step", "one_shot") and filter_classes:
            all_class_labels_raw = [c.get("label") for c in (parsed.get("classes") or []) if c.get("label")]
            # Filter to allowed clinical classes only: avoid passing governance labels into hierarchy prompt.
            _filter_set = set(filter_classes) if filter_classes else set()
            class_labels = [lb for lb in all_class_labels_raw if lb in _filter_set] if _filter_set else all_class_labels_raw
            if not class_labels:
                class_labels = all_class_labels_raw
            # Run hierarchy extraction when at least 2 classes are available.
            # The old trigger-based gating skipped chunks with valid implicit hierarchy.
            if len(class_labels) >= 2:
                phase_h_known = list(class_labels)
                if inject_vocab_guardrails:
                    if allowed_classes and len(phase_h_known) < 5:
                        for gc in (allowed_classes or []):
                            if gc not in phase_h_known:
                                phase_h_known.append(gc)
                    if "Therapy" not in phase_h_known and use_core:
                        phase_h_known = ["Therapy"] + phase_h_known
                    if "Baseline Therapy" not in phase_h_known and use_core:
                        phase_h_known = ["Baseline Therapy"] + phase_h_known
                if examples_hierarchy:
                    hierarchy_prompt = build_phase_prompt(
                        "hierarchy",
                        chunk_text,
                        examples_hierarchy,
                        vocab_and_hints,
                        known_classes=phase_h_known,
                    )
                else:
                    hierarchy_prompt = (
                        "Extract subclass/superclass (is-a) relationships from the text. "
                        "A hierarchy edge means one class is a type, subtype, category member, "
                        "or specific instance of another.\n\n"
                        "Look for: 'such as', 'is a', 'type of', 'include:', 'includes:', "
                        "enumeration lists, category headings, or any pattern where items are "
                        "grouped under a parent category.\n\n"
                        "Use ONLY the following class labels for subClass and superClass. "
                        "Output valid JSON with keys: \"classes\", \"relations\", \"hierarchy\". "
                        "Set \"classes\" and \"relations\" to empty arrays. "
                        "In \"hierarchy\", list objects with \"subClass\", \"superClass\", and \"evidence\". "
                        "Evidence must be an exact verbatim quote from the text.\n"
                        "Example: {\"subClass\": \"Ventilation\", \"superClass\": \"Baseline Therapy\", \"evidence\": \"baseline therapy: ventilation, sedation\"}\n\n"
                        "Class labels to use: " + "; ".join(phase_h_known) + "\n\n"
                        "Text:\n" + chunk_text + "\n\nYour answer:"
                    )
                if prompt_save_dir is not None:
                    (prompt_save_dir / f"prompt_chunk_{idx:04d}_phase_hierarchy.txt").write_text(
                        hierarchy_prompt, encoding="utf-8"
                    )
                raw_h = llm_client.generate(hierarchy_prompt)
                parsed_h = resolve_class_synonyms(parse_output(raw_h))
                parsed_h_filtered = filter_parsed_to_vocabulary(
                    parsed_h,
                    filter_classes,
                    filter_relations or [],
                    require_evidence=True,
                    chunk_text=chunk_text,
                    min_evidence_length=MIN_EVIDENCE_LENGTH_STRICT,
                    relation_domains=relation_domains,
                    require_hierarchy_lexical_cues=False,
                    require_label_in_evidence=require_label_in_evidence,
                )
                extra_hierarchy = [
                    e for e in (parsed_h_filtered.get("hierarchy") or [])
                    if (e.get("evidence") or "").strip()
                ]
                if extra_hierarchy:
                    existing = parsed.get("hierarchy") or []
                    seen = {(_e.get("subClass"), _e.get("superClass")) for _e in existing}
                    for edge in extra_hierarchy:
                        key = (edge.get("subClass"), edge.get("superClass"))
                        if key not in seen and key[0] and key[1]:
                            seen.add(key)
                            existing.append(edge)
                    parsed["hierarchy"] = existing

        # Dedicated regex pass (Improvement #8): extract hierarchy from trigger phrases in chunk text.
        regex_edges = extract_hierarchy_from_text(chunk_text)
        existing_h = parsed.get("hierarchy") or []
        seen_h = {(e.get("subClass"), e.get("superClass")) for e in existing_h if e.get("subClass") and e.get("superClass")}
        for edge in regex_edges:
            key = (edge.get("subClass"), edge.get("superClass"))
            if key not in seen_h and key[0] and key[1]:
                seen_h.add(key)
                existing_h.append(edge)
        parsed["hierarchy"] = existing_h
        # Hierarchy-from-lexical-cues only: keep only edges whose evidence contains a trigger phrase.
        parsed["hierarchy"] = filter_hierarchy_to_lexical_cues(parsed.get("hierarchy") or [])

        if "chunk_id" not in parsed:
            parsed["chunk_id"] = chunk["chunk_id"]
        stratum = "core" if use_core else ("governance" if use_governance else ("provenance" if use_provenance else "core"))
        parsed["stratum"] = stratum
        if progress_callback is not None:
            progress_callback(idx + 1, len(chunks), f"Processed chunk {idx + 1} of {len(chunks)}")
        parsed_chunks.append(parsed)
    return parsed_chunks
