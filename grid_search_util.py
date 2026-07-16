import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def grid_search(datos, param_grid, model_class, preparar_dataset_fn, train_end, fixed_kwargs=None, max_iter=5):
    """Perform a simple and fast random/grid search using a single validation split at the end of the training data.
    datos: list of draws
    param_grid: dict of parameter lists
    model_class: the classifier class
    preparar_dataset_fn: function to prepare X, y
    train_end: index of the end of the available training data
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}
    
    # Use the last 20% (up to 40 draws) of the training window as validation set
    val_size = min(40, max(5, int(train_end * 0.20)))
    split_idx = train_end - val_size
    
    if split_idx <= 10:
        # Fallback if training data is too small
        split_idx = max(5, train_end - 2)
        
    X_train, y_train = preparar_dataset_fn(datos, 0, split_idx)
    X_val, y_val = preparar_dataset_fn(datos, split_idx, train_end)
    
    best_score = -np.inf
    best_params = None
    
    # Sample from hyperparameter grid
    param_iter = ParameterSampler(param_grid, n_iter=max_iter, random_state=42)
    for params in param_iter:
        model = model_class(**{**fixed_kwargs, **params})
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, probs)
            else:
                pred = model.predict(X_val)
                score = accuracy_score(y_val, pred)
        except Exception:
            # In case of any model fit/predict error, skip parameters
            continue
            
        if score > best_score:
            best_score = score
            best_params = params
            
    # Fallback to the first combination in the grid if none succeeded
    if best_params is None:
        best_params = list(ParameterSampler(param_grid, n_iter=1, random_state=42))[0]
        
    return best_params, best_score
