"""Prompt assembly for MMR few-shot strategy (controlled template only)."""

from __future__ import annotations

from pathlib import Path
from typing import List


def _examples_block(examples: List[str]) -> str:
    if not examples:
        return "(No retrieved examples. Check resources/pool_strict_concepts.json and related task pools.)"
    n = len(examples)
    header = f"--- Retrieved examples ({n}) ---"
    return header + "\n\n" + "\n\n---\n\n".join(examples)


def assemble_mmr_prompt_controlled(
    text: str,
    examples: List[str],
    *,
    vocab_and_hints: str = "",
    phase_instruction: str = "",
) -> str:
    """Controlled vocabulary (Piper 2003 Part B clinical only); used for all few-shot strategies.

    The template has four placeholders:
      {{VOCAB_AND_HINTS}}     — vocab guardrails + NER anchor + candidate terms
      {{PHASE_INSTRUCTION}}   — phase-specific extraction instruction
      {{EXAMPLES}}            — retrieved few-shot examples
      {{TEXT}}                — the chunk text to analyze
    """
    template = Path(__file__).parent / "templates" / "mmr_fewshot_controlled.txt"
    prompt = template.read_text(encoding="utf-8")
    return (
        prompt
        .replace("{{VOCAB_AND_HINTS}}", vocab_and_hints)
        .replace("{{PHASE_INSTRUCTION}}", phase_instruction)
        .replace("{{EXAMPLES}}", _examples_block(examples))
        .replace("{{TEXT}}", text)
    )
