"""Prompt assembly for baseline strategy."""

from __future__ import annotations

from .template_loader import load_template, render_template


def assemble_baseline_prompt(text: str) -> str:
    template = load_template("baseline")
    return render_template(template, text=text)
