"""Load and save results for network intrusion detection models."""
import hashlib
import joblib
import numpy as np
import pandas as pd
import skops.io as sio
from pathlib import Path
from typing import Any


def save_results(
    results: dict[str, Any],
    filename: str,
    directory: str | Path,
    verbose: bool = True,
) -> None:
    """
    Save model training and testing results to disk.

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
    

def load_results(
    filename: str | Path, 
    verbose: bool = True
) -> dict:
    """
    Load model training results.
    
    Parameters
    ----------
    filename : str
        Path to results pickle to load
    verbose : bool, default True
        Print information

    Returns
    -------
    dict
        Loaded model training results
    """
    if verbose:
        print('='*70)
        print('Load Results')
        print('-'*70)

    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    results = joblib.load(file_path)

    if verbose:
        print('Results loaded:')
        print(f'{filename}')
        print()

    return results


def save_model(
    bundle: dict, 
    filename: str | Path,
    compresslevel: int = 3,
    verbose: bool = True
) -> str:
    """
    Save model bundle as a skops file for deployment

    Parameters
    ----------
    bundle : dict
        Dictionary containing model artifacts and metadata:
        - 'model': Trained sklearn model or pipeline
        - 'label_encoder': Fitted LabelEncoder
        - 'scaler': Fitted StandardScaler or None
        - 'feature_names': List of feature names
        - 'metadata': Dict of metadata (e.g., metrics)
    filename : str | Path
        Path to save skops file. Should end in .skops
    compresslevel : int, default 3
        Compression level (0-9) (0 disables compression)
    verbose : bool, default True
        Print information
    """
    required_keys = {
        'model',
        'label_encoder',
        'scaler',
        'feature_names',
        'metadata'
    }
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f'bundle is missing required keys: {missing_keys}')

    file_path = Path(filename)
    if file_path.suffix != '.skops':
        raise ValueError(f'filename must end in .skops, not {file_path.suffix}')
    
    if not 0 <= compresslevel <= 9:
        raise ValueError(f'compress must be between 0 and 9, not {compresslevel}')

    _smoke_test(bundle)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(bundle, file_path, compresslevel=compresslevel)

    file_hash = _hash_file(file_path)
    hash_path = file_path.with_suffix('.sha256')
    hash_path.write_text(file_hash)
    
    if verbose:
        print(f'Model saved: {file_path}')
        print(f'SHA256 hash: {file_hash}')

    return file_hash


def _hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def _smoke_test(bundle: dict) -> None:
    feature_names = bundle['feature_names']
    X_dummy = pd.DataFrame(
        np.zeros((1, len(feature_names))), 
        columns=feature_names
    )
    if bundle['scaler'] is not None:
        X_dummy = pd.DataFrame(
            bundle['scaler'].transform(X_dummy),
            columns=feature_names
        )
    try:
        bundle['model'].predict(X_dummy)
    except Exception as e:
        raise ValueError(f'Model failed prediction smoke test: {e}')