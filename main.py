from pprint import pprint

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mainfold_evaluator import GridSearchManifoldEvaluator
from datasets import get_dataset

# 1. Load All Datasets
datasets = get_dataset("all")

# 2. Define the kLDA Pipeline
klda_pipeline = Pipeline([
    ('kpca', KernelPCA(kernel='rbf', fit_inverse_transform=False)),
    ('lda', LinearDiscriminantAnalysis())
])

# 3. Optimized Configuration
models_config = {
    # --- PCA ---
    "PCA": (
        PCA(random_state=42),
        [
            {"n_components": [1, 2], "svd_solver": ["randomized"]},
            {"n_components": [0.95, 0.99, "mle"], "svd_solver": ["full"]}
        ]
    ),

    # --- kPCA ---
    "kPCA": (
        KernelPCA(random_state=42, n_jobs=-1),
        {
            "n_components": [1, 2],
            "kernel": ["rbf", "poly", "cosine"],
            "gamma": [0.1, 1, 10]
        }
    ),

    # --- LDA (Fixing the Crash) ---
    "LDA": (
        LinearDiscriminantAnalysis(),
        [
            # Case 1: SVD Solver (Robust, no shrinkage)
            {
                "solver": ["svd"],
                # n_components must be < n_classes. None = max possible
                "n_components": [None] 
            },
            # Case 2: Eigen Solver (Needs shrinkage to avoid LinAlgError)
            {
                "solver": ["eigen"],
                "shrinkage": ["auto", 0.5], # FIXED: Added shrinkage
                "n_components": [None]
            }
            # DELETED: 'lsqr' solver (It caused the NotImplementedError)
        ]
    ),

    # --- kLDA ---
    "kLDA": (
        klda_pipeline,
        {
            "kpca__n_components": [None], 
            "kpca__kernel": ["rbf", "cosine"],
            "kpca__gamma": [0.1, 1],
            # Only use SVD for the final step to be safe
            "lda__solver": ["svd"], 
        }
    )
}

# 4. Run
gs = GridSearchManifoldEvaluator()
gs.evaluate(datasets, models_config)

pprint(gs.get_metrics_summary())

print("=" * 10)

gs.get_all_trains_df().to_csv('results.csv')

gs.plot_projections()