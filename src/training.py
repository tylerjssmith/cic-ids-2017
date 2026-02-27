"""Train models for network intrusion detection."""
import copy
import importlib
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def split_data(
    df: pd.DataFrame, 
    label_col: str = 'label', 
    test_size: float = 0.2, 
    random_state: int = 76, 
    stratify: bool = True, 
    verbose: bool = True
) -> dict:
    """
    Split data for training and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    label_col : str, default 'label'
        Label column
    test_size : float, default 0.2
        Proportion of data for test set
    random_state : int, default 76
        Random state for train_test_split()
    stratify : bool, default True
        Whether to stratify on label_col
    verbose : bool, default True
        Print information

    Returns
    -------
    dict
        Dictionary containing:
        - X_train (pd.DataFrame)
        - X_test (pd.DataFrame)
        - y_train (pd.Series)
        - y_test (pd.Series)
    """
    if verbose:
        print('='*70)
        print('Split Data')
        print('-'*70)
    
    if label_col not in df.columns:
        raise ValueError(f'Column "{label_col}" not found in df')
    
    X = df.drop(columns=[label_col])
    y = df[label_col]

    if verbose:
        print(f'Test Size:    {test_size}')
        print(f'Random State: {random_state}')
        print(f'Stratify:     {stratify}')
        print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    if verbose:
        # Dataset sizes
        print(f'Dataset Sizes:')
        print(f'Full:     {len(df):>8,} rows')
        train_pcts_size = len(X_train) / len(df) * 100
        print(f'Training: {len(X_train):>8,} rows ({train_pcts_size:.1f}%)')
        test_pcts_size = len(X_test) / len(df) * 100
        print(f'Test:     {len(X_test):>8,} rows ({test_pcts_size:.1f}%)')
        print()

        # Class balance
        full_cnts = y.value_counts().sort_index()
        full_pcts = (full_cnts / len(y) * 100).round(1)
        
        train_cnts = y_train.value_counts().sort_index()
        train_pcts = (train_cnts / len(y_train) * 100).round(1)
        
        test_cnts = y_test.value_counts().sort_index()
        test_pcts = (test_cnts / len(y_test) * 100).round(1)

        max_cls_len = max(len(str(class_val)) for class_val in full_cnts.index)
        class_col_width = max(max_cls_len, 5)
        
        print(f'Class Balance Comparison:')
        print('-'*70)
        print(f'{"Class":<{class_col_width}} {"Full Dataset":>18} '
              f'{"Training Set":>16} {"Test Set":>16}')
        print('-'*70)
        
        for class_val in full_cnts.index:
            full_str = (
                f'{full_cnts[class_val]:>6,} ({full_pcts[class_val]:>4.1f}%)'
                )
            train_str = (
                f'{train_cnts[class_val]:>6,} ({train_pcts[class_val]:>4.1f}%)'
                )
            test_str = (
                f'{test_cnts[class_val]:>6,} ({test_pcts[class_val]:>4.1f}%)'
                )
            
            print(f'{str(class_val):<{class_col_width}} {full_str:>18} '
                  f'{train_str:>16} {test_str:>16}')
        
        print('-'*70)
        
        # Stratification check
        if stratify:
            max_diff = max(
                abs(train_pcts - full_pcts).max(),
                abs(test_pcts - full_pcts).max()
            )
            if max_diff < 0.5:
                print('Success: Class distribution differences <0.5%')
            else:
                print(f'Class distribution difference: {max_diff:.2f}%')
        else:
            print(f'Stratification disabled (stratify={stratify})')
        
        print()

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def train_models(
    data: dict, 
    models: dict,
    filename: str = None,
    directory: str | Path = 'models/',
    cv_k: int = 10, 
    cv_n_jobs: int = 1, 
    use_smote: bool = False,
    smote_max: int = 2500,
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
        - 'feature_importances': DataFrame of feature names and importances,
          or None if the model does not expose them.
        - 'precision_mean': Weighted-average precision across classes (CV).
        - 'precision_std': Std deviation of per-class precision across classes.
        - 'recall_mean': Weighted-average recall across classes (CV).
        - 'recall_std': Std deviation of per-class recall across classes.
        - 'f1_mean': Weighted-average F1 score across classes (CV).
        - 'f1_std': Std deviation of per-class F1 across classes.
    """
    if verbose:
        print('='*70)
        print('Train Models')
        print('-'*70)
    
    X = data['X_train']
    y = data['y_train']
    
    feature_names = X.columns.tolist()

    if cv_kwargs is None:
        cv_kwargs = {}
    cv_kwargs = {k: v for k, v in cv_kwargs.items() if k != 'cv'}

    # Encode Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print('LabelEncoder():')
        for code, label in enumerate(label_encoder.classes_):
            print(f'- {label:<18} -> {code}')
        print()

    # Build SMOTE sampling_strategy
    smote_sampling_dict = {}
    if use_smote:
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

        for label, count in class_counts.items():
            if count < smote_max:
                smote_sampling_dict[label] = smote_max

                if verbose:
                    class_name = label_encoder.inverse_transform([label])[0]
                    print(f'- {class_name:<{max_name_len}} '
                        f'{count:>{max_count_len},} -> {smote_max:,}')

        if verbose:
            print()

    # Train Models
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
        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
            if verbose:
                print('StandardScaler applied')
                print()
        else:
            X_scaled = X

        # Run Cross-validation
        try:
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

            y_pred = cross_val_predict(
                cv_estimator, X_scaled, y_encoded,
                cv=cv_k,
                n_jobs=cv_n_jobs,
                **cv_kwargs
            )

            # Calculate Cross-validation Scores
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

            class_names = label_encoder.inverse_transform(
                np.unique(y_encoded)
            )
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

            result_dict = {
                'label_encoder': copy.deepcopy(label_encoder),
                'scaler': scaler,
                'metrics': scores_df,
                'precision_mean': precision_weighted,
                'precision_std': precision.std(),
                'recall_mean': recall_weighted,
                'recall_std': recall.std(),
                'f1_mean': f1_weighted,
                'f1_std': f1.std()
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

            model.fit(X_final, y_final)
            result_dict['model'] = model

            # Extract Feature Importances
            feature_importances_df = None

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importances_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False, ignore_index=True)

                if verbose:
                    print('Feature Importances (tree-based):')
                    print(
                        feature_importances_df
                            .head(10)
                            .round(4)
                            .to_string(index=False)
                    )
                    print()

            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    importances = np.abs(coef).mean(axis=0)
                else:
                    importances = np.abs(coef)

                feature_importances_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False, ignore_index=True)

                if verbose:
                    print('Feature Importances (coefficient-based):')
                    print(
                        feature_importances_df
                            .head(10)
                            .round(4)
                            .to_string(index=False)
                    )
                    print()

            else:
                if verbose:
                    print('Feature importances are not available.')
                    print()

            result_dict['feature_importances'] = feature_importances_df
            results[model_name] = result_dict

        except Exception as e:
            print(f'Error training {model_name}: {e}')
            continue

    # Save Results
    if filename and directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / filename
        joblib.dump(results, file_path)

        if verbose:
            print('-'*70)
            print('Results saved:')
            print(f'{file_path}')
            print()

    return results