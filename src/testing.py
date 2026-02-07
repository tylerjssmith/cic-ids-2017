"""Evaluate trained models for networking intrusion detection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from input_output import load_data_splits, load_results


def evaluate_models(
    results: dict, 
    data: dict, 
    verbose: bool = True
) -> dict:
    """
    Evaluate models on test data.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by train_models() or loaded by 
        load_results()
    data : dict
        Data splits returned by split_data() or loaded by 
        load_data_splits()
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
        y_predict_encoded = model.predict(X_test_filtered)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded,
            y_predict_encoded,
            labels=np.unique(y_test_encoded),
            zero_division=0
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


def main():
    """Parse command-line arguments and run evaluation."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results',
        type=str,
        required=True
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--quiet',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Load data and results
    results_path = Path(args.results)
    results_dir = str(results_path.parent)
    results_filename = results_path.name
    
    results = load_results(results_dir, results_filename)
    data = load_data_splits(args.data)
    
    # Evaluate models
    evaluation = evaluate_models(results, data, verbose=not args.quiet)
    
    return evaluation


if __name__ == "__main__":
    evaluation = main()