"""Train models for network intrusion detection."""
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
    cv_n_jobs: int = -1, 
    use_smote: bool = False,
    smote_max: int = 10000,
    random_state: int = 76,
    verbose: bool = True,
    cv_kwargs: dict = None,
) -> dict:
    """
    Train models.

    Parameters
    ----------
    data : dict
        Training data
        data should contain 'X_train' and 'y_train'.
        X_train should be a DataFrame with column names.
    models : dict
        Models configuration
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
        Filename to save results
    directory : str | Path, default 'models/'
        Directory to save results
    cv_k : int, default 10
        Number of folds
    cv_n_jobs : int, default -1
        Number of parallel jobs by cross_val_predict()
    use_smote : bool, default False
        Whether to apply SMOTE for class imbalance
    smote_max : int, default 10000
        Max number of samples to generate for small classes
    random_state : int, default 76
        Random state
    verbose : bool, default True
        Print information
    cv_kwargs : dict, default None
        Keyword arguments passed to cross_val_predict()

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping model names to results:
        - 'model': Trained model on full training data
        - 'scaler': StandardScaler (if scale_features=True in model_config)
        - 'label_encoder': LabelEncoder fitted on target labels
        - 'metrics': DataFrame with precision, recall, F1 per class, overall
        - 'feature_importances': DataFrame with feature names and importances
        - 'precision_mean': Mean precision (weighted) from CV
        - 'precision_std': Std deviation per class
        - 'recall_mean': Mean recall (weighted) from CV
        - 'recall_std': Std deviation per class
        - 'f1_mean': Mean F1 score (weighted) from CV
        - 'f1_std': Std deviation per class
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

    # LabelEncoder()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if verbose:
        print('LabelEncoder():')
        for code, label in enumerate(label_encoder.classes_):
            print(f'- {label:<18} -> {code}')
        print()

    # SMOTE()
    if use_smote:
        class_counts = Counter(y_encoded)
        sampling_dict = {}
        for class_label, count in class_counts.items():
            if count < smote_max:
                sampling_dict[class_label] = smote_max

        smote = SMOTE(
            sampling_strategy=sampling_dict, 
            random_state=random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)

        if verbose:
            n_y_pre = len(y_encoded)
            n_y_pst = len(y_resampled)
            n_y_dif = n_y_pst - n_y_pre
            print('-'*70)
            print('SMOTE():')
            print(f'Samples Before: {n_y_pre:,}')
            print(f'Samples After:  {n_y_pst:,} (+{n_y_dif:,}) '
                  f'(smote_max={smote_max:,})')
            print()

    else:
        X_resampled, y_resampled = X, y_encoded
    
    # Models
    results = dict()

    # For each model in models...
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

        # Apply StandardScaler (if scale_features=True in model_config)
        scaler = None
        if scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_resampled)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
            if verbose:
                print('StandardScaler applied')
                print()
        else:
            X_scaled = X_resampled

        # Run Cross-validation
        try:
            y_pred = cross_val_predict(
                model, X_scaled, y_resampled,
                cv=cv_k,
                n_jobs=cv_n_jobs, 
                **cv_kwargs
            )

            # Calculate Scores
            precision, recall, f1, support = precision_recall_fscore_support(
                y_resampled, y_pred, 
                labels=np.unique(y_resampled), 
                zero_division=0
            )

            precision_weighted, recall_weighted, f1_weighted, _ = (
                precision_recall_fscore_support(
                    y_resampled, y_pred, 
                    average='weighted', 
                    zero_division=0
                )
            )

            if verbose:
                print('Cross-validation Scores:')
                print('Weighted Average:')
                print(f'- precision: {precision_weighted:.4f}')
                print(f'- recall:    {recall_weighted:.4f}')
                print(f'- f1_score:  {f1_weighted:.4f}')
                print()

            # Store Results
            class_names = label_encoder.inverse_transform(
                np.unique(y_resampled)
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
            scores_df = pd.concat([scores_df, overall_row], 
                ignore_index=True
            )
            
            if verbose:
                print('Per Class:')
                print(scores_df.round(4).to_string(index=False))
                print()

            result_dict = {
                'label_encoder': label_encoder,
                'scaler': scaler,
                'metrics': scores_df,
                'precision_mean': precision_weighted,
                'precision_std': precision.std(),
                'recall_mean': recall_weighted,
                'recall_std': recall.std(),
                'f1_mean': f1_weighted,
                'f1_std': f1.std()
            }

            # Train Model w/ Full Data
            model.fit(X_scaled, y_resampled)
            result_dict['model'] = model
            
            # Extract Feature Importances
            feature_importances_df = None
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importances_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', 
                    ascending=False, ignore_index=True)
                
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
                }).sort_values('importance', 
                    ascending=False, ignore_index=True)
                
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