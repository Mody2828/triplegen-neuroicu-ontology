"""Experiment configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    prompt_strategy: str
    llm_provider: str
    llm_model: str | None
    corpus_path: str
    gold_standard_path: str | None
    gold_standard_mode: str | None
    gold_corpus_path: str | None


def load_config(path: str | Path) -> Dict[str, Any]:
    from ..prompting.parse import safe_json_loads
    text = Path(path).read_text(encoding="utf-8")
    return safe_json_loads(text)


def load_strategies(config: Dict[str, Any]) -> List[StrategyConfig]:
    strategies = []
    for item in config.get("strategies", []):
        llm_provider = item.get("llm_provider") or config.get("llm_provider")
        if not llm_provider:
            raise ValueError("llm_provider must be specified in config. Supported: 'openai', 'anthropic', 'google', 'groq', 'huggingface'")
        llm_model = item.get("llm_model") or config.get("llm_model")
        if llm_model is not None and not isinstance(llm_model, str):
            llm_model = str(llm_model).strip() or None
        elif isinstance(llm_model, str):
            llm_model = llm_model.strip() or None
        strategies.append(
            StrategyConfig(
                name=item["name"],
                prompt_strategy=item["prompt_strategy"],
                llm_provider=llm_provider,
                llm_model=llm_model,
                corpus_path=item.get("corpus_path", config.get("corpus_path", "data/corpus")),
                gold_standard_path=item.get("gold_standard_path", config.get("gold_standard_path")),
                gold_standard_mode=item.get("gold_standard_mode", config.get("gold_standard_mode")),
                gold_corpus_path=item.get("gold_corpus_path", config.get("gold_corpus_path")),
            )
        )
    return strategies

