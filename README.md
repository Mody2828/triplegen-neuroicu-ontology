# TripleGen: LLM-Driven Ontology Engineering for Neuro-ICU

## Overview

**TripleGen** is a research project focused on using **Large Language Models (LLMs)** to support **ontology engineering** in the **Neuro-ICU** domain.

The project explores how effectively LLMs can reconstruct a **high-quality clinical ontology** from literature in a **low-resource domain**, using **BrainIT and related Neuro-ICU papers** together with a curated gold standard for evaluation. TripleGen combines prompt-based extraction, retrieval-enhanced examples, structured post-processing, and evaluation to study both the strengths and limitations of LLM-driven ontology generation.

## Repository Contents

This repository contains the core documentation, resources, and implementation materials for the project.

### Key Documents

| Document                                       | Description                                                                                                                                                                                                 |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**TripleGen_WepApp.md**](TripleGen_WepApp.md) | A visual guide to the web application, including step-by-step usage instructions and annotated screenshots.                                                                                                 |
| [**Implementation.md**](implementation.md)     | A detailed explanation of the system design and implementation, including pipeline architecture, prompting strategies, operational modes, post-processing layers, evaluation design, and the gold standard. |
| [**Scope.md**](Scope.md)                       | A concise description of the project scope, including what is included, excluded, and prioritised within the research.                                                                                      |

## Project Aim

To design, implement, and evaluate an **LLM-driven ontology engineering framework** tailored to the **Neuro-ICU** domain.

## Research Question

**How effectively can Large Language Models reconstruct a high-quality domain ontology in a low-resource clinical setting?**

## Core Objectives

* Develop a **reproducible pipeline** for ontology extraction, refinement, and evaluation.
* Compare multiple **LLMs**, **prompting strategies**, and **improvement layers**.
* Measure performance using structured evaluation metrics, including **ablation-style comparisons** across pipeline stages.
* Provide an accessible **web-based interface** for running experiments and visualising ontology outputs.

## Deliverables

* A modular ontology engineering pipeline for Neuro-ICU literature.
* Comparative experiments across zero-shot, one-shot, and few-shot strategies.
* Evaluation outputs covering raw extraction, cleanup, and optional enhancement layers.
* A web application with interactive ontology visualisation and experiment controls.

## Getting Started

To understand the project structure and methodology, start with [**Implementation.md**](implementation.md).
This is the best entry point for understanding how TripleGen works, including its pipeline stages, prompting strategies, evaluation approach, and system design.

## Notes

TripleGen is designed as both:

* a **research framework** for investigating LLM-based ontology engineering, and
* a **practical prototype** for experimenting with ontology extraction workflows in the Neuro-ICU domain.

---
