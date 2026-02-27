"""Train models for network intrusion detection."""
import copy
import importlib
import joblib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_models(
    data: dict, 
    models: dict,
    filename: str = None,
    directory: str | Path = 'models/',
    cv_k: int = 10, 
    cv_n_jobs: int = 1, 
    use_smote: bool = False,
    smote_max: int = 2500,
    use_class_weights: bool = False,
    random_state: int = 76,
    verbose: bool = True,
    cv_kwargs: dict = None,
) -> dict:
    """
    Train models.

    Parameters
    ----------
    data : dict
        Training data.
        Must contain 'X_train' and 'y_train'.
        X_train should be a DataFrame with column names.
    models : dict
        Models configuration.
        Expected format:
        {
            'model_name': {
                'module': 'module.path',
                'class': 'ClassName',
                'hyperparameters': {...},
                'scale_features': bool (optional)
            }
        }
    filename : str, default None
        Filename to save results.
    directory : str | Path, default 'models/'
        Directory to save results.
    cv_k : int, default 10
        Number of folds.
    cv_n_jobs : int, default 1
        Number of parallel jobs for cross_val_predict().
    use_smote : bool, default False
        Whether to apply SMOTE for class imbalance. SMOTE is applied inside
        each CV fold via an imblearn Pipeline to prevent data leakage and
        applied to the full training set before fitting the final model.
    smote_max : int, default 2500
        Maximum number of samples per minority class after oversampling.
    use_class_weights : bool, default False
        Whether to compute sample weights and pass them to model.fit().
        Weights are computed using sklearn's compute_sample_weight() with
        class_weight='balanced'. Cannot be True when use_smote is True.
    random_state : int, default 76
        Random state passed to SMOTE. Note: pass random_state inside
        'hyperparameters' to control randomness within each model.
    verbose : bool, default True
        Print information.
    cv_kwargs : dict, default None
        Additional keyword arguments passed to cross_val_predict().
        Note: 'cv' is reserved and will be ignored if passed here.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping model names to results:
        - 'model': Trained model (or pipeline) on full training data.
        - 'scaler': StandardScaler fitted on training data, or None.
          Apply scaler.transform() to X_test before calling model.predict().
        - 'label_encoder': LabelEncoder fitted on target labels.
          Use label_encoder.inverse_transform() to recover class names.
        - 'metrics': DataFrame with precision, recall, F1 per class and
          overall weighted average. CV metrics reflect fold-level performance
          on held-out folds only.
        - 'precision_mean': Weighted-average precision across classes (CV).
        - 'precision_std': Std deviation of per-class precision across classes.
        - 'recall_mean': Weighted-average recall across classes (CV).
        - 'recall_std': Std deviation of per-class recall across classes.
        - 'f1_mean': Weighted-average F1 score across classes (CV).
        - 'f1_std': Std deviation of per-class F1 across classes.
        - 'y_pred_proba': ndarray of shape (n_samples, n_classes) containing
          out-of-fold predicted probabilities from cross-validation. Each row
          reflects predictions made when that sample was in a held-out fold.
          Use with 'y_encoded' for threshold selection and precision-recall
          curve analysis.
        - 'y_encoded': ndarray of true encoded labels corresponding to
          'y_pred_proba'. Use label_encoder.inverse_transform() to recover
          class names.
    """
    if verbose:
        print('='*70)
        print('Train Models')
        print('-'*70)

    if use_smote and use_class_weights:
        raise ValueError(
            'use_smote and use_class_weights cannot both be True. '
            'Use one approach to address class imbalance.'
        )
    
    X = data['X_train']
    y = data['y_train']
    
    feature_names = X.columns.tolist()

    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs = {k: v for k, v in cv_kwargs.items() if k != 'cv'}

    # Encode Labels
    y_encoded, label_encoder = _encode_labels(y, verbose)

    # Build SMOTE
    smote_sampling_dict = {}
    k_neighbors = 5
    if use_smote:
        smote_sampling_dict, k_neighbors = _build_smote_strategy(
            y_encoded=y_encoded,
            label_encoder=label_encoder,
            smote_max=smote_max,
            cv_k=cv_k,
            verbose=verbose,
        )

    # For Model in Models...
    results = dict()

    for model_name, model_config in models.items():
        if verbose:
            print('-'*70)
            print(model_name)
            print('-'*70)
        
        # Parse Configuration
        module_name = model_config.get('module')
        class_name = model_config.get('class')
        hyperparameters = model_config.get('hyperparameters', {})
        scale_features = model_config.get('scale_features', False)
        
        if not module_name or not class_name:
            print(f'Error: Missing module or class for {model_name}')
            continue

        # Initialize Model
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            model = model_class(**hyperparameters)
                
        except (ImportError, AttributeError) as e:
            print(f'Error loading {model_name}: {e}')
            continue
        except TypeError as e:
            print(f'Error initializing {model_name} with hyperparameters: {e}')
            continue

        # Apply StandardScaler
        scaler, X_scaled = _apply_scaler(
            X, 
            feature_names,
            scale_features,
            verbose
        )

        # Train Models
        try:
            # Score Model via Cross-validation
            y_pred_proba, scores_df, score_summary = _compute_cv_scores(
                model=model,
                X_scaled=X_scaled,
                y_encoded=y_encoded,
                label_encoder=label_encoder,
                use_smote=use_smote,
                smote_sampling_dict=smote_sampling_dict,
                k_neighbors=k_neighbors,
                random_state=random_state,
                cv_k=cv_k,
                cv_n_jobs=cv_n_jobs,
                cv_kwargs=cv_kwargs,
                verbose=verbose,
            )

            precision = score_summary['precision']
            recall = score_summary['recall']
            f1 = score_summary['f1']
            precision_weighted = score_summary['precision_weighted']
            recall_weighted = score_summary['recall_weighted']
            f1_weighted = score_summary['f1_weighted']

            result_dict = {
                'label_encoder': copy.deepcopy(label_encoder),
                'scaler': scaler,
                'metrics': scores_df,
                'precision_mean': precision_weighted,
                'precision_std': precision.std(),
                'recall_mean': recall_weighted,
                'recall_std': recall.std(),
                'f1_mean': f1_weighted,
                'f1_std': f1.std(),
                'y_pred_proba': y_pred_proba,
                'y_encoded': y_encoded
            }

            # Train Final Model
            if use_smote:
                smote_final = SMOTE(
                    sampling_strategy=smote_sampling_dict,
                    k_neighbors=k_neighbors,
                    random_state=random_state
                )
                X_final, y_final = smote_final.fit_resample(X_scaled, y_encoded)
                X_final = pd.DataFrame(X_final, columns=feature_names)

                if verbose:
                    n_before = len(y_encoded)
                    n_after = len(y_final)
                    print(f'SMOTE() applied for final fit:')
                    print(f'  Samples before: {n_before:,}')
                    print(f'  Samples after:  {n_after:,} '
                          f'(+{n_after - n_before:,})')
                    print()
            else:
                X_final, y_final = X_scaled, y_encoded

            if use_class_weights:
                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=y_final
                )
            else:
                sample_weights = None

            model.fit(X_final, y_final, sample_weight=sample_weights)
            result_dict['model'] = model
            results[model_name] = result_dict

        except Exception as e:
            print(f'Error training {model_name}: {e}')
            continue

    if filename and directory:
        _save_results(results, filename, directory, verbose)

    return results


# --- Helper Functions --------------------------------------------------------
def _apply_scaler(
    X: pd.DataFrame,
    feature_names: list[str],
    scale_features: bool,
    verbose: bool,
) -> tuple:
    """
    Optionally fit and apply a StandardScaler to X.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_names : list[str]
        Column names, used to reconstruct DataFrame after scaling.
    scale_features : bool
        Whether to apply scaling.
    verbose : bool
        Print information.

    Returns
    -------
    tuple[StandardScaler | None, pd.DataFrame]
        - scaler: Fitted StandardScaler, or None if scaling was not applied.
        - X_scaled: Scaled feature matrix as DataFrame, or original X.
    """
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        if verbose:
            print('StandardScaler applied')
            print()
        return scaler, X_scaled

    return None, X


def _build_smote_strategy(
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    smote_max: int,
    cv_k: int,
    verbose: bool,
) -> tuple[dict, int]:
    """
    Build SMOTE sampling strategy and compute safe k_neighbors.

    Parameters
    ----------
    y_encoded : np.ndarray
        Encoded target labels.
    label_encoder : LabelEncoder
        Fitted label encoder for class name lookup.
    smote_max : int
        Maximum number of samples per minority class after oversampling.
    cv_k : int
        Number of CV folds, used to compute safe k_neighbors.
    verbose : bool
        Print information.

    Returns
    -------
    tuple[dict, int]
        - sampling_dict: Maps encoded class label to target sample count.
        - k_neighbors: Safe number of neighbors for SMOTE given fold size.

    Raises
    ------
    ValueError
        If the smallest minority class has too few samples for SMOTE.
    """
    if verbose:
        print('SMOTE():')

    class_counts = Counter(y_encoded)

    min_count = min(
        count for count in class_counts.values() if count < smote_max
    )
    min_count_in_fold = int(min_count * (cv_k - 1) / cv_k)
    k_neighbors = min(5, min_count_in_fold - 1)

    if k_neighbors < 1:
        raise ValueError(
            f'Smallest minority class has too few samples ({min_count}) '
            f'to use SMOTE with cv_k={cv_k}. Reduce cv_k or disable SMOTE.'
        )

    if verbose:
        max_name_len = max(
            len(label_encoder.inverse_transform([label])[0])
            for label in class_counts
        )
        max_count_len = max(
            len(f'{count:,}')
            for count in class_counts.values()
        )
        if k_neighbors < 5:
            print(f'k_neighbors reduced to {k_neighbors} '
                  f'(min class count = {min_count})')

    sampling_dict = {}
    for label, count in class_counts.items():
        if count < smote_max:
            sampling_dict[label] = smote_max

            if verbose:
                class_name = label_encoder.inverse_transform([label])[0]
                print(f'- {class_name:<{max_name_len}} '
                      f'{count:>{max_count_len},} -> {smote_max:,}')

    if verbose:
        print()

    return sampling_dict, k_neighbors


def _compute_cv_scores(
    model,
    X_scaled: pd.DataFrame,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    use_smote: bool,
    smote_sampling_dict: dict,
    k_neighbors: int,
    random_state: int,
    cv_k: int,
    cv_n_jobs: int,
    cv_kwargs: dict,
    verbose: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Run cross-validation and compute per-class and weighted average scores.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    X_scaled : pd.DataFrame
        Feature matrix, scaled if applicable.
    y_encoded : np.ndarray
        Encoded target labels.
    label_encoder : LabelEncoder
        Fitted label encoder for recovering class names.
    use_smote : bool
        Whether to wrap the model in a SMOTE pipeline for CV.
    smote_sampling_dict : dict
        Sampling strategy passed to SMOTE.
    k_neighbors : int
        Number of neighbors for SMOTE.
    random_state : int
        Random state for SMOTE.
    cv_k : int
        Number of CV folds.
    cv_n_jobs : int
        Number of parallel jobs for cross_val_predict().
    cv_kwargs : dict
        Additional keyword arguments for cross_val_predict().
    verbose : bool
        Print information.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame, dict]
        - y_pred_proba: Out-of-fold predicted probabilities,
          shape (n_samples, n_classes).
        - scores_df: Per-class and weighted average metrics as a DataFrame.
        - score_summary: Dict of per-class arrays and weighted scalars:
          precision, recall, f1, precision_weighted, recall_weighted,
          f1_weighted.
    """
    if use_smote:
        smote = SMOTE(
            sampling_strategy=smote_sampling_dict,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        cv_estimator = ImbPipeline([
            ('smote', smote),
            ('model', model)
        ])
    else:
        cv_estimator = model

    y_pred_proba = cross_val_predict(
        cv_estimator, X_scaled, y_encoded,
        cv=cv_k,
        n_jobs=cv_n_jobs,
        method='predict_proba',
        **cv_kwargs
    )
    y_pred = np.argmax(y_pred_proba, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_encoded, y_pred,
        labels=np.unique(y_encoded),
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_encoded, y_pred,
            average='weighted',
            zero_division=0
        )
    )

    if verbose:
        print('Weighted Average Scores (CV):')
        print(f'- precision: {precision_weighted:.4f}')
        print(f'- recall:    {recall_weighted:.4f}')
        print(f'- f1_score:  {f1_weighted:.4f}')
        print()

    class_names = label_encoder.inverse_transform(np.unique(y_encoded))
    scores_df = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })
    overall_row = pd.DataFrame({
        'class': ['overall_weighted'],
        'precision': [precision_weighted],
        'recall': [recall_weighted],
        'f1_score': [f1_weighted],
        'support': [support.sum()]
    })
    scores_df = pd.concat([scores_df, overall_row], ignore_index=True)

    if verbose:
        print('Per-Class Scores:')
        print(scores_df.round(4).to_string(index=False))
        print()

    score_summary = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    return y_pred_proba, scores_df, score_summary


def _encode_labels(
    y: pd.Series, 
    verbose: bool
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode labels.

    Parameters
    ----------
    y : pd.Series
        Labels to encode.
    verbose : bool
        Print information.

    Returns
    -------
    tuple[np.ndarray, LabelEncoder]
        - y_encoded: Encoded labels as integer array
        - label_encoder: LabelEncoder fitted on y
          Use label_encoder.inverse_transform() to recover class names.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print('LabelEncoder():')
        for code, label in enumerate(label_encoder.classes_):
            print(f'- {label:<18} -> {code}')
        print()
    
    return y_encoded, label_encoder


def _save_results(
    results: dict[str, Any],
    filename: str,
    directory: str | Path,
    verbose: bool,
) -> None:
    """
    Save training results to disk using joblib.

    Parameters
    ----------
    results : dict
        Training results to save.
    filename : str
        Filename for the saved file.
    directory : str | Path
        Directory to save the file in.
    verbose : bool
        Print information.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename
    joblib.dump(results, file_path)

    if verbose:
        print('-'*70)
        print('Results saved:')
        print(f'{file_path}')
        print()