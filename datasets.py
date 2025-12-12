import pandas as pd # Make sure pandas is imported
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_circles, make_moons, make_blobs, load_digits, load_wine, load_iris
from ucimlrepo import fetch_ucirepo 

def get_dataset(name, n_samples=1000):
    if name == "all":
        # Returns a dict of everything
        return {
            k: get_dataset(k, n_samples) 
            for k in ["swiss_roll", "s_curve", "circles", "moons", "blobs", "digits", "wine", "iris", "ionosphere"]
        }

    # --- Manifolds (Discretized for Classification) ---
    if name == "swiss_roll":
        X, t = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
        # FIX: Use qcut to ensure equal class sizes
        y = pd.qcut(t, q=8, labels=False) 
        return X, y

    elif name == "s_curve":
        X, t = make_s_curve(n_samples=n_samples, noise=0.1, random_state=42)
        # FIX: Use qcut to ensure equal class sizes
        y = pd.qcut(t, q=8, labels=False)
        return X, y

    elif name == "circles":
        return make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=42)
    elif name == "moons":
        return make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif name == "blobs":
        return make_blobs(n_samples=n_samples, centers=3, random_state=42)
    elif name == "digits":
        d = load_digits()
        return d.data, d.target
    elif name == "wine":
        d = load_wine()
        return d.data, d.target
    elif name == "iris":
        d = load_iris()
        return d.data, d.target
    elif name == "ionosphere":
        # fetch dataset 
        ionosphere = fetch_ucirepo(id=52) 
        
        # data (as pandas dataframes) 
        X = ionosphere.data.features 
        y = ionosphere.data.targets
        return X, y
    elif name == "isolet":
        # fetch dataset 
        isolet = fetch_ucirepo(id=54) 
        
        # data (as pandas dataframes) 
        X = isolet.data.features 
        y = isolet.data.targets
        return X, y
    else:
        raise ValueError(f"Unknown: {name}")