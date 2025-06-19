## Evaluating-Positional-Analog-Scanning-as-a-Method-for-Lead-Optimization
Guide: Dr. Arjun Ray, Dr. Jacob Kongsted

Screening the ChEMBL database to find analog pairs that have been tested in the same assay. This allows us to explore the increments in binding affinity by adding functional groups to a molecule, while comparing them to other substituents. This will prioritize molecules for experimental testing and lead optimization.


# Key Methods:

#Feature Engineering
Morgan Fingerprints and external descriptors (e.g., logP, HAC) used to represent molecules.

Variance Filtering and Hierarchical Clustering (Silhouette Score ~0.66 for ~621 clusters) to reduce redundancy.

# Feature Selection Techniques
Mutual Information + Sequential Forward Selection (MISFS).

PCA — retained 95% variance with 66 components.

ANOVA, RFE, and Boruta — used as baselines for statistical and model-based selection.

# Modeling Approaches
Reformulated the task as a binary classification problem (Improved vs. Reduced potency).

# Evaluated models:

Traditional ML: Random Forest, XGBoost

AutoML: H2O AutoML

Deep Learning: AutoKeras, Supervised Autoencoders

# Evaluation Metrics
Accuracy, ROC-AUC, Silhouette Scores (for clustering), and SHAP values for feature interpretability.

#Results
SHAP analysis identified key molecular descriptors influencing predictions.

PCA + AutoKeras was the most promising pipeline for performance and interoperability.


