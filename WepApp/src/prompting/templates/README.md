# Prompt Templates

Templates used per strategy. When you add or change rules (e.g. hierarchy, evidence, vocabulary), check the templates for the strategies you use.

## Shared structure (all strategies)

- **SYSTEM ROLE** — Medical ontology extraction engine; Neuro-ICU / BrainIT focus.
- **CORE CONSTRAINTS** — Source fidelity, evidence anchor, hierarchy rules, scope limitation (no metadata/boilerplate).
- **OUTPUT FORMAT** — JSON only: `classes`, `relations`, `hierarchy` with `evidence` (exact fragment) per item. **Strict benchmark templates** (`one_shot.json`, `mmr_fewshot_controlled.txt`) use **label + evidence only** (no `definition`); allowed relation labels are explicitly listed (includes, has_source, treats, has_target, secondary_to, is_a). List rule: do not create relations from list co-occurrence. Hierarchy rule: no narrative fragments (“good example”, “important factor”, etc.); only clean noun-phrase subclass/superclass pairs.
- **TEXT TO ANALYZE** — Delimiter and chunk text (and, for one-shot/few-shot, retrieved examples before it).

## Strategy → template(s)

| Strategy | Template | Pool(s) |
|----------|----------|---------|
| **Baseline** | `baseline.json` | — |
| **One-shot** | `one_shot.json` | `pool_strict_concepts.json` (1 example) |
| **Few-shot I** | `mmr_fewshot_controlled.txt` | `pool_strict_concepts.json` (3 examples) |
| **Few-shot II** | `mmr_fewshot_controlled.txt` | Phase 1: concept pool; Phase 2: relation pool |
| **Few-shot III** | `mmr_fewshot_controlled.txt` | Phase 1: concept; Phase 2: relation; Phase 3: hierarchy pool |

Phased prompts (few-shot II/III) are built in `run.py` via `build_phase_prompt()`; they use the same base text as `mmr_fewshot_controlled.txt` with phase-specific instructions.

## Files in this directory

- `baseline.json` — Zero-shot (constrained) prompt.
- `one_shot.json` — One-shot prompt (single concept example).
- `mmr_fewshot_controlled.txt` — Few-shot I/II/III (concept/relation/hierarchy task pools).

## What to check when updating

- **Output format** — All templates ask for JSON with `classes`, `relations`, `hierarchy`; strict ones use label + evidence only (no definition).
- **Examples** — One-shot/few-shot use placeholders `{{EXAMPLES}}`/`{examples}` and `{{TEXT}}`/`{text}` for injected examples and chunk text.
- **Task pools** — See `docs/task_pool_wiring.md` for pool wiring.
