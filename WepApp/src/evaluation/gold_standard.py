"""Gold standard adapter: JSON and TTL (RDF) support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..prompting.parse import safe_json_loads


def load_gold_standard(path: str | Path) -> Dict:
    """Load gold standard from JSON or TTL/RDF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gold standard path not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".ttl", ".rdf", ".owl", ".n3"):
        return _load_gold_from_ttl(path)
    return safe_json_loads(path.read_text(encoding="utf-8"))


def _load_gold_from_ttl(path: Path) -> Dict:
    """Parse TTL with rdflib and convert to internal {classes, relations, hierarchy} format."""
    try:
        from rdflib import Graph
        from rdflib.namespace import OWL, RDF, RDFS
    except ImportError:
        raise RuntimeError(
            "Install rdflib to load TTL gold standard: pip install rdflib "
            "(or use a project venv: python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt)"
        )

    g = Graph()
    g.parse(path, format="turtle")

    classes: List[Dict] = []
    seen_labels: set = set()

    # Collect owl:Class and rdfs:Class subjects (rdf:type is the correct predicate)
    for s in g.subjects(predicate=RDF.type, object=OWL.Class):
        label = _get_label(g, s)
        if label and label not in seen_labels:
            seen_labels.add(label)
            definition = _get_comment(g, s)
            classes.append({"label": label, "definition": definition or None, "synonyms": []})
    for s in g.subjects(predicate=RDF.type, object=RDFS.Class):
        label = _get_label(g, s)
        if label and label not in seen_labels:
            seen_labels.add(label)
            definition = _get_comment(g, s)
            classes.append({"label": label, "definition": definition or None, "synonyms": []})

    # Object properties as relations (domain/range as class labels)
    relations: List[Dict] = []
    for s in g.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
        prop_label = _get_label(g, s)
        domain = _get_domain_class_label(g, s)
        range_ = _get_range_class_label(g, s)
        if prop_label and domain and range_:
            relations.append({"label": prop_label, "domain": domain, "range": range_, "definition": None})

    # Hierarchy: rdfs:subClassOf (subclass -> superclass)
    # A class may have multiple superclasses; iterate all objects per subject.
    hierarchy: List[Dict] = []
    seen_hier: set = set()
    for subj in set(g.subjects(predicate=RDFS.subClassOf)):
        sub_label = _get_label(g, subj)
        if not sub_label:
            continue
        for super_class_node in g.objects(subject=subj, predicate=RDFS.subClassOf):
            super_label = _get_label(g, super_class_node)
            if super_label and (sub_label, super_label) not in seen_hier:
                seen_hier.add((sub_label, super_label))
                hierarchy.append({"subClass": sub_label, "superClass": super_label})

    return {"classes": classes, "relations": relations, "hierarchy": hierarchy}


def _get_label(g, node) -> str | None:
    from rdflib.namespace import RDFS
    for o in g.objects(subject=node, predicate=RDFS.label):
        return str(o) if o else None
    # Fallback: local name of URI (e.g. neuroicu:Patient -> Patient, or ...#Patient -> Patient)
    s = str(node) if node else ""
    if "#" in s:
        return s.split("#")[-1]
    if "/" in s:
        return s.split("/")[-1]
    return None


def _get_comment(g, node) -> str | None:
    from rdflib.namespace import RDFS
    for o in g.objects(subject=node, predicate=RDFS.comment):
        return str(o) if o else None
    return None


def _get_domain_class_label(g, prop_node) -> str | None:
    from rdflib.namespace import RDFS
    for o in g.objects(subject=prop_node, predicate=RDFS.domain):
        return _get_label(g, o)
    return None


def _get_range_class_label(g, prop_node) -> str | None:
    from rdflib.namespace import RDFS
    for o in g.objects(subject=prop_node, predicate=RDFS.range):
        return _get_label(g, o)
    return None
