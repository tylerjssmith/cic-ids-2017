"""Train final model for network intrusion detection on full dataset."""
import argparse
import pandas as pd
from pathlib import Path
import skops.io as sio
from input_output import load_data_splits, load_results

def finalize_model(
    results: dict, 
    model_name: str, 
    data: dict, 
    filename: str, 
    directory: str | Path = 'models/', 
    verbose: bool = True
):
    """
    Train final model on full dataset.

    Parameters
    ----------
    results : dict
        Dictionary returned by train_models()
        The function expects the following keys under model_name:
        model, label_encoder, scaler (if scaling was used)
    model_name : str
        Dictionary key for model
    data : dict
        Dictionary containing data
        The function expects the following keys:
        X_train, X_test, y_train, y_test
    filename : str
        File name for saving trained model in directory
    directory : str | Path, default 'models/'
        Where to save trained model
    verbose : bool, default True
        Print information
    
    Returns
    -------
    dict
        Dictionary with:
        - model: Trained model
        - label_encoder: Fitted label encoder
        - scaler: Fitted scaler (or None if scaling was not used)
        - feature_names: List of feature names
        - metadata
    """
    if verbose:
        print('-'*70)
        print('Finalize Model')
        print('-'*70)

    X = pd.concat([data['X_train'], data['X_test']], 
        axis=0, ignore_index=True, copy=False)
    y = pd.concat([data['y_train'], data['y_test']], 
        axis=0, ignore_index=True, copy=False)

    model_results = results[model_name]
    label_encoder = model_results['label_encoder']
    scaler = model_results.get('scaler', None)

    if not hasattr(label_encoder, 'classes_'):
        raise ValueError("Label encoder has not been fitted yet")

    y_encoded = label_encoder.transform(y)

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        if verbose:
            print("StandardScaler refitted on full dataset.")

    else:
        X_scaled = X

    model = model_results['model']
    model.fit(X_scaled, y_encoded)

    metadata = {
        'model_name': model_name,
        'model_class': type(model).__name__,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'n_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_),
        'scaling_used': scaler is not None,
        'training_date': pd.Timestamp.now().isoformat(),
    }

    final_package = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'scaler': scaler,
        'metadata': metadata
    }

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if not filename.endswith('.skops'):
        filename = f"{filename}.skops"

    file_path = directory / filename
    sio.dump(final_package, file_path)

    if verbose:
        print(f'Model:          {metadata["model_name"]}')
        print(f'Class:          {metadata["model_class"]}')
        print(f'Samples:        {metadata["n_samples"]:,}')
        print(f'Features:       {metadata["n_features"]}')
        print(f'Classes:        {metadata["n_classes"]}')
        print(f'Scaling used:   {metadata["scaling_used"]}')
        print()
        print('Package saved:')
        print(file_path)
        print()

    return final_package


def main():
    """Parse command-line arguments and finalize model."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True
    )
    parser.add_argument(
        '--quiet',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    results_dir = str(results_path.parent)
    results_filename = results_path.name
    
    output_path = Path(args.output)
    output_dir = (
        str(output_path.parent) 
        if output_path.parent != Path('.') 
        else 'models'
    )    
    output_filename = output_path.name
    
    results = load_results(results_dir, results_filename)
    data = load_data_splits(args.data)

    final_model = finalize_model(
        results=results, 
        model_name=args.model, 
        data=data, 
        directory=output_dir,
        filename=output_filename,
        verbose=not args.quiet
    )
    
    return final_model


if __name__ == "__main__":
    final_model = main()