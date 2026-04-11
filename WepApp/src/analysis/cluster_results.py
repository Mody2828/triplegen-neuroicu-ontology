"""Cluster extracted ontology classes using semantic embeddings + Ward's hierarchical clustering.

Designed for use from the Flask UI — takes classes from ontology.json,
returns JSON-serializable results for Chart.js visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_model():
    """Lazy-load the sentence-transformers model (singleton via retrieval.embed)."""
    from src.retrieval.embed import _get_model
    return _get_model()


def _auto_label(classes: List[Dict], labels: np.ndarray, n_clusters: int) -> Dict[int, str]:
    """Extract top TF-IDF terms per cluster to auto-name them."""
    cluster_docs = {i: [] for i in range(n_clusters)}
    for i, c in enumerate(classes):
        text = f"{c.get('label', '')} {c.get('definition', '')}".lower()
        cluster_docs[int(labels[i])].append(text)

    mega = [" ".join(cluster_docs[i]) for i in range(n_clusters)]

    stops = ['patient', 'patients', 'data', 'clinical', 'used', 'using',
             'based', 'study', 'monitoring', 'brain', 'care', 'intensive',
             'unit', 'measured', 'defined', 'refers', 'value', 'values',
             'assessment', 'associated', 'analysis', 'information', 'specific',
             'related', 'various', 'may', 'also', 'can', 'one', 'two',
             'within', 'provides', 'represents', 'involves', 'process',
             'system', 'condition', 'measure', 'level', 'levels', 'type',
             'including', 'management', 'collected', 'collection']

    try:
        vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words=stops,
                              min_df=1, sublinear_tf=True)
        X_tfidf = vec.fit_transform(mega).toarray()
        features = vec.get_feature_names_out()
        names = {}
        for i in range(n_clusters):
            top_idx = X_tfidf[i].argsort()[-3:][::-1]
            terms = [features[j] for j in top_idx]
            names[i] = " / ".join(t.title() for t in terms)
        return names
    except Exception:
        return {i: f"Cluster {i}" for i in range(n_clusters)}


def cluster_classes(classes: List[Dict]) -> Dict[str, Any]:
    """Cluster ontology classes and return JSON-serializable results.

    Args:
        classes: List of class dicts with at least 'label' and optionally 'definition', 'evidence'.

    Returns:
        Dict with keys: clusters, summary, pca_points, silhouette_score, optimal_k, total_classes
    """
    if len(classes) < 5:
        return {"error": "Need at least 5 classes to cluster", "total_classes": len(classes)}

    # Build sentences for embedding — richer context produces better clusters
    sentences = []
    for c in classes:
        label = c.get("label", "")
        definition = (c.get("definition") or "").strip()
        evidence = (c.get("evidence") or "").strip()

        parts = [label]
        if definition:
            parts.append(definition)
        elif evidence:
            # No definition available — use evidence as context so the embedding
            # captures more than just the bare label
            parts.append(f"A clinical concept described as: {evidence[:200]}")
        if evidence:
            parts.append(evidence[:200])
        sentences.append(". ".join(parts))

    # Encode
    model = _get_model()
    X = model.encode(sentences, show_progress_bar=False, batch_size=64)

    # Silhouette analysis for optimal k
    max_k = min(25, len(classes) - 1)
    min_k = min(5, max_k)
    k_range = range(min_k, max_k + 1)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        km_labels = km.fit_predict(X)
        s = silhouette_score(X, km_labels, metric='cosine')
        scores.append(s)

    best_k = list(k_range)[int(np.argmax(scores))]

    # Ward's hierarchical clustering
    agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = agg.fit_predict(X)
    final_sil = float(silhouette_score(X, labels, metric='cosine'))

    # Auto-label clusters
    cluster_names = _auto_label(classes, labels, best_k)

    # PCA for 2D scatter
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    variance = [float(v) for v in pca.explained_variance_ratio_]

    # Build response
    pca_points = []
    for i, c in enumerate(classes):
        pca_points.append({
            "x": float(X2[i, 0]),
            "y": float(X2[i, 1]),
            "label": c.get("label", ""),
            "cluster": int(labels[i]),
            "cluster_name": cluster_names[int(labels[i])],
        })

    cluster_list = []
    for cid in sorted(cluster_names.keys()):
        members = sorted([c.get("label", "") for i, c in enumerate(classes) if labels[i] == cid])
        cluster_list.append({
            "id": cid,
            "name": cluster_names[cid],
            "size": len(members),
            "members": members,
        })

    # Silhouette curve for chart
    sil_curve = [{"k": int(k), "score": float(s)} for k, s in zip(k_range, scores)]

    return {
        "total_classes": len(classes),
        "optimal_k": best_k,
        "silhouette_score": round(final_sil, 4),
        "pca_variance": variance,
        "pca_points": pca_points,
        "clusters": cluster_list,
        "silhouette_curve": sil_curve,
    }
