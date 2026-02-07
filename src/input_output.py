"""Load and save data and models for network intrusion detection models."""
import pandas as pd
import joblib
from pathlib import Path


def load_data(
    directory: str | Path, 
    filenames: list = None, 
    clean_names: bool = True, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load one or more CSV files.

    Parameters
    ----------
    directory : str | Path
        Where to read CSV files
    filenames : list, default None
        List of CSV files in directory to load
        If None, all CSV files in directory will be loaded.
    clean_names : bool, default True
        Format column names
        If True, strips leading and trailing whitespace, makes lowercase, 
        and uses underscores for spaces and slashes.
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame, optionally with clean column names
    """
    if verbose:
        print('='*70)
        print('Load Data')
        print('-'*70)

    directory = Path(directory)

    if verbose:
        print('Directory:')
        print(directory)
        print()

    if not directory.exists():
        raise ValueError(f'Directory not found: {directory}')
    
    if not directory.is_dir():
        raise ValueError(f'Path is not a directory: {directory}')
    
    if verbose:
        print('Files:')

    if not filenames:
        filenames = sorted([f.name for f in directory.glob('*.csv')])

    if not isinstance(filenames, list):
        raise TypeError(f'filenames argument must be a list')

    data = []

    for filename in filenames:
        path = Path(filename)
        file_path = path if path.is_absolute() else directory / filename

        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        df = pd.read_csv(file_path)
        data.append(df)

        if verbose:
            rows = len(df)
            cols = len(df.columns)
            print(file_path.name)
            print(f'({rows:,} rows, {cols:,} columns)')
            print()
    
    data = pd.concat(data, axis=0, ignore_index=True)

    if clean_names:
        data.columns = data.columns\
            .str.strip()\
            .str.lower()\
            .str.replace(' ', '_')\
            .str.replace('/', '_')
        
        if verbose:
            print('Cleaned column names.')
            print()

    if verbose:
        rows = len(data)
        cols = len(data.columns)
        memory = data.memory_usage(deep=True).sum() / 1_000_000
        print('-'*70)
        print(f'Loaded Rows:    {rows:,}')
        print(f'Loaded Columns: {cols:,}')
        print(f'Memory:         {memory:,.2f} MB')
        print()

    return data


def save_data_splits(
    data: dict, 
    directory: str | Path,
    prefix: str = '',
    verbose: bool = True
) -> dict:
    """
    Save data splits.

    Parameters
    ----------
    data : dict
        Input data splits
    directory : str | Path
        Where to save data splits
    prefix : str, default ''
        File name prefix
    verbose : bool, default True
        Print information

    Returns
    -------
    dict
        Input data splits
    """
    if verbose:
        print('='*70)
        print('Save Data Splits')
        print('-'*70)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = prefix + '_'

    if verbose:
        print('Directory:')
        print(f'{directory}')
        print()
        print('Files:')

    for k, v in data.items():
        filename = f'{prefix}{k}.parquet'
        file_path = directory / filename

        if isinstance(v, pd.DataFrame):
            v.to_parquet(file_path)
        elif isinstance(v, pd.Series):
            v.to_frame().to_parquet(file_path)
        else:
            raise TypeError(f'{k} is not a DataFrame or Series')
        
        if verbose:
            print(f'- {filename}')

    if verbose:
        print()

    return data


def load_data_splits(
    directory: str | Path, 
    X_train: str = 'X_train.parquet',
    X_test: str = 'X_test.parquet',
    y_train: str = 'y_train.parquet',
    y_test: str = 'y_test.parquet',
    verbose: bool = True
) -> dict:
    """
    Load data splits.

    Parameters
    ----------
    directory : str | Path
    X_train : str, default 'X_train.parquet'
        Filename for training features
    X_test : str, default 'X_test.parquet'
        Filename for test features
    y_train : str, default 'y_train.parquet'
        Filename for training labels
    y_test : str, default 'y_test.parquet'
        Filename for test labels
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
        print('Load Data Splits')
        print('-'*70)

    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f'Directory not found: {directory}')

    if verbose:
        print('Directory:')
        print(f'{directory}')
        print()
        print('Files:')

    files = {'X_train': X_train, 'X_test': X_test, 
             'y_train': y_train, 'y_test': y_test}
    
    for name, filename in files.items():
        file_path = directory / filename
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

    data = {
        'X_train': pd.read_parquet(directory / X_train),
        'X_test': pd.read_parquet(directory / X_test),
        'y_train': pd.read_parquet(directory / y_train),
        'y_test': pd.read_parquet(directory / y_test)
    }

    if verbose:
        for k, v in data.items():
            print(f'{k}')                
            print(f'({len(v):,} rows, {len(v.columns):,} columns)')
            print()
    
    # Coerce y_train, y_test to pd.Series (one-dimensional)
    data['y_train'] = data['y_train'].iloc[:, 0]
    data['y_test'] = data['y_test'].iloc[:, 0]

    if verbose:
        print('y_train, y_test coerced to pd.Series')
        print()

    return data


def load_results(
    directory: str | Path, 
    filename: str, 
    verbose: bool = True
) -> dict:
    """
    Load model training results.
    
    Parameters
    ----------
    directory : str | Path
        Where to read results
    filename : str
        File in directory to load
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

    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f'Directory not found: {directory}')
    
    if verbose:
        print('Directory:')
        print(f'{directory}')
        print()

    file_path = directory / filename

    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    results = joblib.load(file_path)

    if verbose:
        print('File:')
        print(f'{filename}')
        print()

    return results