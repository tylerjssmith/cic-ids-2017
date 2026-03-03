"""Train final model for network intrusion detection on full dataset."""
import pandas as pd
from pathlib import Path
from data.input_output import load_data_splits
from models.input_output import load_results

def finalize_model(
    results: str | Path, 
    data: str | Path, 
    model_name: str,   
    verbose: bool = True
):
    """
    Train final model on full dataset.

    Parameters
    ----------
    results : str | Path
        Path to results pickle to be passed
        to load_results()
    data : str | Path
        Directory containing data splits to be passed 
        to load_data_splits()
    model_name : str
        Dictionary key for model
    verbose : bool, default True
        Print information
    
    Returns
    -------
    dict
        Dictionary containing model artifacts and metadata:
        - 'model': Trained sklearn model or pipeline
        - 'label_encoder': Fitted LabelEncoder
        - 'scaler': Fitted StandardScaler or None
        - 'feature_names': List of feature names
        - 'metadata': Dict of metadata (e.g., metrics)
    """
    data = load_data_splits(data)
    results = load_results(results)

    if verbose:
        print('='*70)
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
            print()

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

    if verbose:
        print(f'Model:          {metadata["model_name"]}')
        print(f'Class:          {metadata["model_class"]}')
        print(f'Samples:        {metadata["n_samples"]:,}')
        print(f'Features:       {metadata["n_features"]}')
        print(f'Classes:        {metadata["n_classes"]}')
        print(f'Scaling used:   {metadata["scaling_used"]}')
        print()

    return final_package