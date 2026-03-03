"""Evaluate trained models for networking intrusion detection."""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from data.input_output import load_data_splits
from models.input_output import load_results


def evaluate_models(
    results: str | Path, 
    data: str | Path,
    verbose: bool = True
) -> dict:
    """
    Evaluate models on test data.
    
    Parameters
    ----------
    results : str | Path
        Path to results pickle to be passed
        to load_results()
    data : str | Path
        Directory containing data splits to be passed 
        to load_data_splits()
    verbose : bool, default True
        Print information
    
    Returns
    -------
    dict
        Dictionary mapping model names to evaluation DataFrames
        Each DataFrame contains precision, recall, F1, and support
        per class plus overall weighted averages.
    """
    if verbose:
        print('='*70)
        print('Evaluate Models')
        print('-'*70)
    
    data = load_data_splits(data)
    results = load_results(results)

    if verbose:
        print('Models:')
        for model_name in results.keys():
            print(f'- {model_name}')
        print()

    X_test = data['X_test']
    y_test = data['y_test']

    evaluation = dict()

    for model_name, model_results in results.items():
        if verbose:
            print('-'*70)
            print(model_name)
            print('-'*70)

        label_encoder = model_results['label_encoder']

        unseen_classes = set(y_test) - set(label_encoder.classes_)
        if unseen_classes:
            print(f'Warning: Test set has unseen classes: {unseen_classes}')
            mask = y_test.isin(label_encoder.classes_)
            X_test_filtered = X_test[mask]
            y_test_filtered = y_test[mask]
            y_test_encoded = label_encoder.transform(y_test_filtered)
        else:
            X_test_filtered = X_test
            y_test_encoded = label_encoder.transform(y_test)

        model = model_results['model']
        scaler = model_results['scaler']

        X_input = (
            scaler.transform(X_test_filtered)
            if scaler is not None
            else X_test_filtered
        )

        proba = model.predict_proba(X_input)
        y_predict_encoded = np.argmax(proba, axis=1)

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_test_encoded,
                y_predict_encoded,
                labels=np.unique(y_test_encoded),
                zero_division=0
            )
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_test_encoded, 
                y_predict_encoded, 
                average='weighted', 
                zero_division=0
            )
        )

        class_names = label_encoder.inverse_transform(
            np.unique(y_test_encoded)
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
            print(scores_df.round(4).to_string(index=False))
            print()

        evaluation[model_name] = scores_df

    return evaluation