"""
Train models for network intrusion detection system.
"""

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def train_classifier_cv(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model,
    cv: int = 10,
    scale_features: bool = False,
    verbose: bool = True
):
    """
    Train any scikit-learn binary classifier with cross-validation 
    evaluation.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    model : sklearn estimator
        Any scikit-learn binary classifier (e.g., LogisticRegression, 
        RandomForestClassifier, XGBClassifier)
    cv : int, default 10
        Number of cross-validation folds
    scale_features : bool, default False
        Whether to standardize features (recommended for linear models,
        not necessary for tree-based models)
    verbose : bool, default True
        Print results summary
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Fitted model (trained on full X_train)
        - 'scaler': Fitted StandardScaler (None if scale_features=False)
        - 'cv_results': Cross-validation scores
        - 'cv_mean': Mean scores across folds
        - 'cv_std': Standard deviation across folds
    """
    # Get model name
    model_name = model.__class__.__name__
    
    # Scale features if scale_features=True
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
    else:
        X_train_processed = X_train
    
    # Perform cross-validation
    if verbose:
        print("="*70)
        print(f"Training {model_name} with Cross-Validation")
        print("="*70)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Cross-validation folds: {cv}")
        print(f"Feature scaling: {'Yes' if scale_features else 'No'}")
        print(f"Class balance: {dict(pd.Series(y_train).value_counts())}")
        print(f"\nRunning cross-validation...")
    
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        clone(model),
        X_train_processed,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Calculate means and stds
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
    
    # Print results
    if verbose:
        print("\n" + "="*70)
        print(f"Cross-Validation Results ({cv}-Fold)")
        print("="*70)
        print(f"Precision: {cv_mean['precision']:.4f} (± {cv_std['precision']:.4f})")
        print(f"Recall:    {cv_mean['recall']:.4f} (± {cv_std['recall']:.4f})")
        print(f"F1-Score:  {cv_mean['f1']:.4f} (± {cv_std['f1']:.4f})")
        
        print(f"\nPer-Fold Results:")
        print("-"*70)
        for fold in range(cv):
            print(f"Fold {fold+1:2d}: "
                  f"Precision={cv_results['test_precision'][fold]:.4f}, "
                  f"Recall={cv_results['test_recall'][fold]:.4f}, "
                  f"F1={cv_results['test_f1'][fold]:.4f}")
        print("="*70)
    
    # Train final model on full training set
    model.fit(X_train_processed, y_train)
    
    if verbose:
        print(f"\nFinal {model_name} trained on full training set.")
    
    # Return results
    return {
        'model': model,
        'scaler': scaler,
        'cv_results': cv_results,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'model_name': model_name
    }