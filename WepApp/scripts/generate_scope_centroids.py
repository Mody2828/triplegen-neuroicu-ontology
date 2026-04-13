"""Generate clinical and administrative centroid vectors for the embedding scope fallback.

Encodes representative clinical and administrative passages using all-MiniLM-L6-v2,
computes the mean vector for each category, and saves them as a JSON file in resources/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Representative clinical passages drawn from BrainIT domain papers.
# These should be clearly clinical — physiological monitoring, treatment, patient data.
CLINICAL_PASSAGES = [
    "Intracranial pressure monitoring is a cornerstone of neurointensive care management for severe traumatic brain injury patients.",
    "Cerebral perfusion pressure is calculated as the difference between mean arterial pressure and intracranial pressure.",
    "The Glasgow Coma Scale score on admission was recorded for all patients alongside pupil reactivity.",
    "Continuous minute-by-minute monitoring of heart rate, blood pressure, and intracranial pressure was performed.",
    "Sedation levels were assessed hourly using standardised clinical scales during the ICU stay.",
    "Secondary insults including systemic hypotension and intracranial hypertension were recorded automatically.",
    "Microdialysis catheters were used to measure brain tissue biochemistry including lactate and pyruvate ratios.",
    "Ventilation parameters including tidal volume and respiratory rate were continuously recorded.",
    "Fluid input and output were documented to maintain adequate cerebral perfusion pressure targets.",
    "CT scan findings at admission showed diffuse axonal injury in the majority of severe TBI patients.",
    "Brain temperature was monitored via an intraparenchymal probe placed alongside the ICP sensor.",
    "Vasopressor therapy was initiated when mean arterial pressure dropped below 80 mmHg.",
    "The therapy intensity level quantifies the aggressiveness of treatment for raised intracranial pressure.",
    "Decompressive craniectomy was performed in patients with refractory intracranial hypertension.",
    "Autoregulation status was assessed using the pressure reactivity index PRx derived from ICP and ABP.",
    "Blood gas analysis showed PaO2 and PaCO2 values outside target ranges during secondary insult episodes.",
    "Patient demographics including age, sex, and mechanism of injury were recorded at admission.",
    "The Glasgow Outcome Scale Extended was used to assess functional outcome at six months post injury.",
    "SjO2 monitoring provided continuous measurement of jugular venous oxygen saturation.",
    "Nutritional support was provided via enteral feeding within 48 hours of admission to the neuro-ICU.",
    "Haematology and biochemistry laboratory values were sampled daily during the acute phase.",
    "Cerebral perfusion pressure variability was associated with worse neurological outcomes.",
    "PbtO2 sensors measured partial pressure of brain tissue oxygen to guide oxygen therapy.",
    "Episodes of hypotension defined as systolic blood pressure below 90 mmHg were identified.",
    "Clinical condition at discharge was documented using standard neurological assessment scales.",
]

# Representative administrative passages — governance, methodology, publishing, stats.
ADMIN_PASSAGES = [
    "The BrainIT steering group met quarterly to review data contributing centres and membership guidelines.",
    "Ethics approval was obtained from the institutional review board at each participating centre.",
    "The study was conducted in accordance with the Helsinki Declaration on ethical research principles.",
    "Statistical analysis was performed using IBM SPSS version 25 with significance set at p < 0.05.",
    "A multivariate logistic regression model was fitted to identify independent predictors of outcome.",
    "The receiver operating characteristic curve was used to assess the discriminative ability of the model.",
    "Mann-Whitney U tests were applied for non-normally distributed continuous variables.",
    "The authors declare no competing financial interests or personal relationships.",
    "Supplementary material is available online at the journal homepage.",
    "This study was funded by a grant from the European Commission Framework Programme.",
    "The BrainIT group is a non-profit organisation coordinating multi-centre data collection.",
    "Publication criteria require approval from the steering group and all contributing investigators.",
    "Data validation was performed by trained nurses at each contributing centre using standardised forms.",
    "The grid middleware infrastructure enabled secure data transfer between hospital firewalls.",
    "Informed consent was obtained from the next of kin for all patients enrolled in the study.",
    "Cross-validation with leave-one-out was used to estimate generalisation performance of the classifier.",
    "The Bayesian artificial neural network was trained using Monte Carlo simulation methods.",
    "Kaplan-Meier survival analysis was performed to compare mortality between treatment groups.",
    "Contents lists available at ScienceDirect with links to the journal homepage.",
    "Springer-Verlag Berlin Heidelberg. All rights reserved. Printed on acid-free paper.",
    "The corresponding author is responsible for communicating with the journal on behalf of all co-authors.",
    "Sample size analysis indicated a minimum of 200 patients were required for adequate statistical power.",
    "Bonferroni correction was applied to adjust for multiple comparisons across the outcome variables.",
    "The AVERT-IT project developed web-services for real-time data acquisition from bedside monitors.",
    "Regional coordinators were responsible for training data contributing members at satellite centres.",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "resources" / "scope_centroids.json"


def main() -> None:
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(CLINICAL_PASSAGES)} clinical passages...")
    clinical_vecs = model.encode(CLINICAL_PASSAGES, show_progress_bar=False, batch_size=64)
    clinical_centroid = np.mean(clinical_vecs, axis=0)

    print(f"Encoding {len(ADMIN_PASSAGES)} administrative passages...")
    admin_vecs = model.encode(ADMIN_PASSAGES, show_progress_bar=False, batch_size=64)
    admin_centroid = np.mean(admin_vecs, axis=0)

    # Sanity check: centroids should be different
    cos_sim = float(np.dot(clinical_centroid, admin_centroid) / (
        np.linalg.norm(clinical_centroid) * np.linalg.norm(admin_centroid)
    ))
    print(f"Cosine similarity between centroids: {cos_sim:.4f}")
    if cos_sim > 0.95:
        print("WARNING: centroids are very similar — passages may need more differentiation")

    # Verify on the source passages
    correct_clinical = 0
    for vec in clinical_vecs:
        sim_c = float(np.dot(vec, clinical_centroid) / (np.linalg.norm(vec) * np.linalg.norm(clinical_centroid)))
        sim_a = float(np.dot(vec, admin_centroid) / (np.linalg.norm(vec) * np.linalg.norm(admin_centroid)))
        if sim_c > sim_a:
            correct_clinical += 1

    correct_admin = 0
    for vec in admin_vecs:
        sim_c = float(np.dot(vec, clinical_centroid) / (np.linalg.norm(vec) * np.linalg.norm(clinical_centroid)))
        sim_a = float(np.dot(vec, admin_centroid) / (np.linalg.norm(vec) * np.linalg.norm(admin_centroid)))
        if sim_a > sim_c:
            correct_admin += 1

    print(f"Clinical passages correctly classified: {correct_clinical}/{len(CLINICAL_PASSAGES)}")
    print(f"Admin passages correctly classified: {correct_admin}/{len(ADMIN_PASSAGES)}")

    data = {
        "model": "all-MiniLM-L6-v2",
        "dim": 384,
        "clinical_centroid": clinical_centroid.tolist(),
        "admin_centroid": admin_centroid.tolist(),
        "clinical_passage_count": len(CLINICAL_PASSAGES),
        "admin_passage_count": len(ADMIN_PASSAGES),
        "centroid_cosine_similarity": cos_sim,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"Centroids saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
