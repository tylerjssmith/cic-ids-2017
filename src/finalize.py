import skops.io as sio
import joblib
from typing import Union
from pathlib import Path
import pandas as pd


def finalize_model(
    models_file: Union[str, Path],
    data_file: Union[str, Path],
    model_name: str,
    model_dir : str,
    model_file : str,
    skops_trusted: list,
    verbose: bool = True
) -> dict:
    """
    Train model on full dataset.
    
    Parameters
    ----------
    models_file : str
        Path to skops file containing trained models
    data_file : str
        Path to pickle file containing data splits
    model_name : str
        Model key in models_file to train
    model_dir : str, default None
        Directory to save trained models
    model_file : str, default None
        Filename to save trained models
    skops_trusted : list
        Trusted skops objects
    verbose : bool, default True
        Print information
    """
    if verbose:
        print("="*70)
        print("Finalize Model")
        print("-"*70)

    # Load model
    models_file = Path(models_file)
    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    trained_models = sio.load(models_file, trusted=skops_trusted)
    model_object = trained_models[model_name]
    model = model_object['pipeline']

    # Load data
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = joblib.load(data_file)

    X = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
    y = pd.concat([data['y_train'], data['y_test']], ignore_index=True)
    
    if verbose:
        print(f"Loaded models from: {models_file}")
        print(f"Loaded data from: {data_file}")
        print()

    # Encode labels
    label_encoder = model_object['label_encoder']
    y_encoded = label_encoder.transform(y)

    # Train model
    if verbose:
        print(f"Training {model_name}")
    
    model.fit(X, y_encoded)
    
    if verbose:
        print(f"Training completed")
        print()

    # Prepare model package to save
    model_package = {
        'pipeline': model,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'classes': list(label_encoder.classes_),
        'n_samples': len(X),
        'model_name': model_name
    }

    # Save model
    if model_dir is not None and model_file is not None:
        output_path = Path(model_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / f"{model_file}.skops"
        sio.dump(model_package, model_path)
        
        if verbose:
            print(f"Model saved to: {model_path}")
            print()
        
        return {
            'model': model_package,
            'save_path': str(model_path)
        }
    else:
        if verbose:
            print("Model not saved (model_dir or model_file not specified)")
            print()
        
        return {
            'model': model_package,
            'save_path': None
        }