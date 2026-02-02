"""
Train models in machine learning training pipeline.
"""
import importlib
import skops.io as sio
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_models(
    data: dict,
    models_config: dict,
    cv: int = 10,
    cv_n_jobs: int = 1,
    model_dir: str = None,
    use_mlflow: bool = False,
    mlflow_name: str = None,
    verbose: bool = True
) -> dict:
    """
    Train multiple models with optional MLflow tracking.
    
    Parameters
    ----------
    data : dict
        Dictionary with X_train, y_train
    models_config : dict
        Model configurations
    cv : int, default 10
        Number of folds for cross-validation
    cv_n_jobs : int, default 1
        Number of cores for cross-validation
    model_dir : str, default None
        Directory to save trained models
    use_mlflow : bool, default False
        Whether to use MLflow tracking
    mlflow_name : str, optional
        MLflow experiment name (required if use_mlflow=True)
    verbose : bool, default True
        Print information
    """
    if use_mlflow and mlflow_name is None:
        raise ValueError(
            "mlflow_name is required when use_mlflow=True"
        )

    X_train = data['X_train']
    y_train = data['y_train']
    
    if use_mlflow:
        mlflow.set_experiment(mlflow_name)
    
    trained_models = {}
    
    for model_name, model_config in models_config.items():
        if use_mlflow:
            mlflow.start_run(run_name=model_name)
        
        try:
            if verbose:
                print("="*70)
                print(f"Train Model: {model_name}")
                print("-"*70)
            
            # Get Configuration
            module_path = model_config.get('module')
            module = importlib.import_module(module_path)
            model_class = getattr(module, model_config['class'])
            hyperparams = model_config.get('hyperparameters', {})

            # MLflow: Configuration
            if use_mlflow:
                mlflow.log_param("model_class", model_config['class'])
                mlflow.log_param("module", module_path)
                mlflow.log_param(
                    "scale_features", 
                    model_config.get('scale_features', False)
                )
                mlflow.log_param("cv_folds", cv)
                
                # MLflow: Hyperparameters
                for key, value in hyperparams.items():
                    mlflow.log_param(f"hp_{key}", value)

            model = model_class(**hyperparams)
            
            if model_config.get('scale_features', False):
                scaler = StandardScaler()
                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('classifier', model)
                ])
            else:
                scaler = None
                pipeline = Pipeline([
                    ('classifier', model)
                ])

            if verbose:
                print("Model:")
                print(f"- {'Module:':<20} {module_path}")
                print(f"- {'Class:':<20} {model_config['class']}")
                print()

                print("Hyperparameters (set explicitly):")
                for k, v in hyperparams.items():
                    print(f"- {k:<20} {v}")
                print()
                print(f"Scale Features: {model_config.get('scale_features')}")
                print("-"*70)

            # Cross-validate
            model = model_class(**hyperparams)

            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=['precision', 'recall', 'f1'],
                n_jobs=cv_n_jobs
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
            
            # MLflow: Cross-validation, Summary
            if use_mlflow:
                mlflow.log_metric("cv_precision_mean", cv_mean['precision'])
                mlflow.log_metric("cv_recall_mean", cv_mean['recall'])
                mlflow.log_metric("cv_f1_mean", cv_mean['f1'])
                mlflow.log_metric("cv_precision_std", cv_std['precision'])
                mlflow.log_metric("cv_recall_std", cv_std['recall'])
                mlflow.log_metric("cv_f1_std", cv_std['f1'])
            
            if verbose:
                print(f"Cross-validation Results ({cv}-Fold):")
                print(
                    f"Precision: {cv_mean['precision']:.4f} "
                    f"(± {cv_std['precision']:.4f})"
                )
                print(
                    f"Recall:    {cv_mean['recall']:.4f} "
                    f"(± {cv_std['recall']:.4f})"
                )
                print(
                    f"F1-Score:  {cv_mean['f1']:.4f} "
                    f"(± {cv_std['f1']:.4f})"
                )
                print()
                
                print(f"Per-Fold Results:")
                for fold in range(cv):
                    print(
                        f"Fold {fold+1:2d}: "
                        f"Precision={cv_results['test_precision'][fold]:.4f}, "
                        f"Recall={cv_results['test_recall'][fold]:.4f}, "
                        f"F1={cv_results['test_f1'][fold]:.4f}"
                    )
                print()

            # Train Model
            pipeline.fit(X_train, y_train)
            
            # MLflow: Save Model
            if use_mlflow:
                mlflow.sklearn.log_model(
                    pipeline, 
                    serialization_format="skops",
                    artifact_path="model",
                    skops_trusted_types=[
                        'xgboost.core.Booster', 
                        'xgboost.sklearn.XGBClassifier'
                    ]
                )
            
            # Local: Save Model
            if model_dir is not None:
                output_path = Path(model_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                model_path = output_path / f"{model_name}.skops"
                sio.dump(pipeline, model_path)
                
                # MLflow: Path to Local Save
                if use_mlflow:
                    mlflow.log_param("local_model_path", str(model_path))

            trained_models[model_name] = {
                'pipeline': pipeline,
                'model': model,
                'scaler': scaler,
                'feature_names': X_train.columns.tolist(),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'mlflow_run_id': (
                    mlflow.active_run().info.run_id if use_mlflow else None
                )
            }
        
        finally:
            if use_mlflow:
                mlflow.end_run()
    
    return trained_models


def get_feature_importance(model_objects, model_name):
    """Get feature importances sorted by importance."""
    model = model_objects[model_name]
    classifier = model['pipeline']['classifier']
    return pd.DataFrame({
        'feature': model['feature_names'],
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)

