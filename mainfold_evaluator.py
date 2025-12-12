import pandas as pd
import numpy as np
from joblib import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class GridSearchManifoldEvaluator:
    def __init__(self):
        # Stores the SINGLE best result per (Dataset, Algo)
        self.results = []
        # Stores EVERY result from the GridSearch (High detail)
        self.all_history = [] 
        self.projections = {}
        self.best_estimators = {}

    def evaluate(self, datasets, models_config):
        """
        datasets: dict { 'name': (X, y) }
        models_config: dict { 'name': (ModelInstance, param_grid) }
        """
        self.results = []
        self.all_history = []
        self.projections = {}
        self.best_estimators = {}

        print(f"Grid Searching over {len(datasets)} datasets...")

        for d_name, (X, y) in datasets.items():
            for m_name, (model, params) in models_config.items():
                
                # 1. Create Pipeline
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reducer", model),
                    ("knn", KNeighborsClassifier(n_neighbors=1)),
                ])

                # 2. Adjust Param Grid
                if isinstance(params, list):
                    pipe_params = []
                    for p_dict in params:
                        pipe_params.append({f"reducer__{k}": v for k, v in p_dict.items()})
                else:
                    pipe_params = {f"reducer__{k}": v for k, v in params.items()}

                # 3. Dynamic CV Splitter
                is_continuous = (len(np.unique(y)) > 50) and (y.dtype.kind in "fc")
                cv_splitter = (
                    KFold(n_splits=5, shuffle=True, random_state=42)
                    if is_continuous
                    else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                )

                # 4. Configure Grid
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=pipe_params,
                    scoring="accuracy",
                    cv=cv_splitter,
                    n_jobs=-1,
                )

                try:
                    # 5. Run Search
                    # On Windows the loky resource tracker can raise errors
                    # in some joblib/loky versions. Use the threading backend
                    # to avoid spawning additional processes when possible.
                    with parallel_backend('threading'):
                        grid.fit(X, y)

                    # --- A. Store Best Result (Summary) ---
                    best_mean = grid.best_score_
                    best_index = grid.best_index_
                    best_std = grid.cv_results_["std_test_score"][best_index]

                    self.results.append({
                        "Dataset": d_name,
                        "Algorithm": m_name,
                        "Mean Accuracy": best_mean,
                        "Std Dev": best_std,
                        "Best Params": grid.best_params_,
                    })

                    self.best_estimators[(d_name, m_name)] = grid.best_estimator_

                    # --- B. Store ALL Trains (Detailed History) ---
                    cv_res = grid.cv_results_
                    # Iterate over every combination tested
                    for i in range(len(cv_res['params'])):
                        # Create a base record
                        record = {
                            "Dataset": d_name,
                            "Algorithm": m_name,
                            "Mean Accuracy": cv_res['mean_test_score'][i],
                            "Std Dev": cv_res['std_test_score'][i],
                            "Rank": cv_res['rank_test_score'][i]
                        }
                        # Unpack parameters into columns (removing 'reducer__' prefix for cleanliness)
                        for p_key, p_val in cv_res['params'][i].items():
                            clean_key = p_key.replace("reducer__", "")
                            record[clean_key] = p_val
                        
                        self.all_history.append(record)

                    # 6. Extract Projection
                    best_scaler = grid.best_estimator_.named_steps["scaler"]
                    best_reducer = grid.best_estimator_.named_steps["reducer"]
                    X_scaled = best_scaler.transform(X)
                    X_r = best_reducer.transform(X_scaled)
                    self.projections[(d_name, m_name)] = (X_r, y)

                except Exception as e:
                    print(f"    ! Error on {d_name} with {m_name}: {e}")

    def get_metrics_summary(self):
        """Returns the pivot table of the BEST scores only."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        df["Score"] = df.apply(
            lambda row: f"{row['Mean Accuracy']:.3f} ± {row['Std Dev']:.3f}", axis=1
        )
        return df.pivot(index="Dataset", columns="Algorithm", values="Score")

    def get_all_trains_df(self):
        """Returns a flat DataFrame of EVERY configuration tested."""
        if not self.all_history:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(self.all_history)
        
        # Reorder columns to put identifiers first
        cols = ['Dataset', 'Algorithm', 'Mean Accuracy', 'Std Dev', 'Rank']
        # Add the parameter columns (the rest of the keys)
        param_cols = [c for c in df.columns if c not in cols]
        
        return df[cols + param_cols]

    def plot_projections(self, save_path="./plots"):
        import os
        os.makedirs(save_path, exist_ok=True)
        for (d_name, m_name), (X_r, y) in self.projections.items():
            if X_r.shape[1] < 2:
                print(f"Skipping {d_name} {m_name}: only {X_r.shape[1]} components")
                continue
            plt.figure(figsize=(8,6))
            try:
                y_plot = y.astype(float)
            except (ValueError, TypeError):
                le = LabelEncoder()
                y_plot = le.fit_transform(y)
            scatter = plt.scatter(X_r[:,0], X_r[:,1], c=y_plot, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f"{m_name} on {d_name}")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig(f"{save_path}/{d_name}_{m_name}.png")
            plt.close()