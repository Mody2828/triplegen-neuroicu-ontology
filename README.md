# TripleGen: LLM-Driven Ontology Engineering for Neuro-ICU

## Overview

**TripleGen** is a research project focused on using **Large Language Models (LLMs)** to support **source-faithful ontology engineering** in the **Neuro-ICU** domain.

The project investigates how effectively LLMs can reconstruct and refine a domain ontology from **Neuro-ICU literature**, particularly in a **low-resource clinical setting** where high-quality ontologies are difficult to build manually. TripleGen combines prompt-based extraction, retrieval-enhanced examples, structured post-processing, and evaluation to explore both the strengths and limitations of LLM-driven ontology generation.

## Project Aim

To design, implement, and evaluate **TripleGen**, an LLM-driven framework for **source-faithful ontology engineering** in the **Neuro-ICU domain**.

## Main Research Question

**How effectively can Large Language Models support source-faithful ontology engineering in the low-resource Neuro-ICU domain?**

## Research Objectives

* Develop a reproducible pipeline for **ontology extraction, refinement, and evaluation** from Neuro-ICU literature.
* Compare multiple **prompting strategies**, including zero-shot, one-shot, and phased few-shot approaches.
* Evaluate the contribution of **retrieval-based examples** to extraction quality.
* Assess the impact of **post-processing layers**, including cleanup, schema-guided completion, and reasoning.
* Measure ontology quality using evidence-based and structure-aware metrics such as **precision, recall, evidence fidelity, hierarchy quality, and ontology consistency**.
* Explore how extraction performance varies across **different paper types and ontology extraction scenarios**.
* Provide a **web-based interface** for running experiments and visualising ontology outputs.

## Repository Contents

This repository contains the core documentation, resources, and implementation materials for TripleGen.

### Key Documents

| Document                                       | Description                                                                                                                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**TripleGen_WepApp.md**](TripleGen_WepApp.md) | A visual guide to the web application, including step-by-step usage instructions and annotated screenshots.                                                                                       |
| [**Implementation.md**](implementation.md)     | A detailed explanation of the system design and implementation, including pipeline architecture, prompting strategies, retrieval setup, cleanup layers, evaluation design, and the gold standard. |
| [**Scope.md**](Scope.md)                       | A concise description of the project scope, including what is included, excluded, and prioritised within the research.                                                                            |

## Core Deliverables

* A reproducible **LLM-driven ontology engineering pipeline** for Neuro-ICU literature.
* A set of **prompting strategies** for strict and task-specific ontology extraction.
* Retrieval-based workflows for **one-shot and few-shot extraction**.
* Post-processing components for **cleanup, completion, and reasoning-based refinement**.
* A comparative experimental framework for analysing strategies, models, and improvement layers.
* A web application with ontology visualisation, experiment controls, and exportable outputs.

## Project Structure

TripleGen is designed as both:

* a **research framework** for studying LLM-based ontology engineering, and
* a **practical prototype** for running ontology extraction experiments in the Neuro-ICU domain.

The system supports:

* literature ingestion and preprocessing,
* scope-aware chunking and retrieval,
* ontology extraction with multiple prompting strategies,
* optional refinement layers,
* and structured evaluation of ontology quality.

## Getting Started

To understand the project structure and methodology, start with [**Implementation.md**](implementation.md).
This is the best entry point for understanding how TripleGen works, including its pipeline stages, prompting strategies, operational modes, and evaluation workflow.

For project boundaries and research framing, see [**Scope.md**](Scope.md).
For application usage, see [**TripleGen_WepApp.md**](TripleGen_WepApp.md).

## Notes

TripleGen is intended to investigate not only whether LLMs can extract ontology elements, but also **how different prompting, retrieval, and refinement strategies affect the quality, faithfulness, and usefulness of the resulting ontology**.

---
