"""
Train models in machine learning training pipeline.
"""
import importlib
import joblib
from pathlib import Path

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

def train_models(
    data: dict, 
    models_config: str, 
    cv: int = 10,
    output_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Train multiple models.
    
    Parameters
    ----------
    data : dict
        Dictionary with X_train, y_train
    models_config : dict
        Model configurations
    cv : int, default 10
        Number of folds for cross-validation
    output_dir : str, default None
        Directory to save trained models
    verbose : bool
        Print training progress
    """    
    X_train = data['X_train']
    y_train = data['y_train']
    
    trained_models = {}
    
    for model_name, model_config in models_config.items():
        if verbose:
            print("="*70)
            print(f"Train Model: {model_name}")
            print("-"*70)
        
        # Get Configuration
        module_path = model_config.get('module')
        module = importlib.import_module(module_path)
        model_class = getattr(module, model_config['class'])
        
        hyperparams = model_config.get('hyperparameters', {})

        # Scale Features
        X_train_processed = X_train.copy()
        
        scaler = None
        if model_config.get('scale_features', False):
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)

        if verbose:
            print("Model:")
            print(f"- {'Module:':<20} {model_config.get('module')}")
            print(f"- {'Class:':<20} {model_config.get("class")}")
            print()

            print("Hyperparameters:")
            for k, v in hyperparams.items():
                print(f"- {k:<20} {v}")
            print()
            print(f"Scale Features: {model_config.get("scale_features")}")
            print("-"*70)

        # Cross-validate
        model = model_class(**hyperparams)

        cv_results = cross_validate(
            model,
            X_train_processed,
            y_train,
            cv=cv,
            scoring=['precision', 'recall', 'f1'],
            n_jobs=-1
        )
        
        cv_mean = {
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'f1': cv_results['test_f1'].mean()
        }
        
        cv_std = {
            'precision': cv_results['test_precision'].std(),
            'recall': cv_results['test_recall'].std(),
            'f1': cv_results['test_f1'].std()
        }
        
        if verbose:
            print(f"Cross-validation Results ({cv}-Fold):")
            print(f"Precision: {cv_mean['precision']:.4f} (± {cv_std['precision']:.4f})")
            print(f"Recall:    {cv_mean['recall']:.4f} (± {cv_std['recall']:.4f})")
            print(f"F1-Score:  {cv_mean['f1']:.4f} (± {cv_std['f1']:.4f})")
            print()
            
            print(f"Per-Fold Results:")
            for fold in range(cv):
                print(f"Fold {fold+1:2d}: "
                      f"Precision={cv_results['test_precision'][fold]:.4f}, "
                      f"Recall={cv_results['test_recall'][fold]:.4f}, "
                      f"F1={cv_results['test_f1'][fold]:.4f}")
            print()

        # Train Model
        model.fit(X_train_processed, y_train)
        
        # Save Model
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            model_path = output_path / f"{model_name}.pkl"
            joblib.dump({'model': model, 'scaler': scaler}, model_path)

        trained_models[model_name] = {
            'model': model,
            'scaler': scaler,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
    
    return trained_models