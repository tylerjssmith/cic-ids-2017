"""Process data for network intrusion detection models."""
import copy
import numpy as np
import pandas as pd 
import re
import warnings


def drop_features(
    df: pd.DataFrame, 
    drop: list = None, 
    zero_variance: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop features from DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop : list, default None
        List of column names to drop
    zero_variance : bool, default True
        Whether to drop columns with zero variance
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with features removed
    """
    if verbose:
        print('='*70)
        print('Drop Features')
        print('-'*70)

    df = df.copy()
    n_initial = len(df.columns)

    if drop:
        not_found = [col for col in drop if col not in df.columns]
        if not_found:
            raise ValueError(f'{not_found} not found in df')
    
        df.drop(columns=drop, inplace=True)

        if verbose:
            n_drop = len(drop)
            print(f'Dropped {n_drop} named columns:')
            for col in drop:
                print(f'- {col}')
            print()

    if zero_variance:
        num_cols = df.select_dtypes(include=[np.number]).columns
        zv = df[num_cols].var()[df[num_cols].var() == 0].index.tolist()
        df.drop(columns=zv, inplace=True)

        if verbose:
            n_zv = len(zv)
            print(f'Dropped {n_zv} columns with zero variance:')
            for col in zv:
                print(f'- {col}')
            print()

    if verbose:
        n_final = len(df.columns)
        print('-'*70)
        print(f'Columns Before: {n_initial}')
        print(f'Columns After:  {n_final}')
        print()

    return df


def keep_features(
    data: dict, 
    keep: list, 
    X_keys: list = ['X_train', 'X_test'],
    list_features: bool = True,
    verbose: bool = True
) -> dict:
    """
    Keep features in data splits (dropping all others).
    
    Parameters
    ----------
    data : dict
        Input data splits
    keep : list
        List of column names to keep
    X_keys : list, default ['X_train','X_test']
        List of keys in data to process
    list_features : bool, default True
        If True, and verbose is True, lists features kept after
        dropping other features
    verbose : bool, default True
        Print information
    
    Returns
    -------
    dict
        Data splits with features kept
    """
    if verbose:
        print('-'*70)
        print('Keep Features')
        print('-'*70)

    _data = copy.deepcopy(data)
    
    for key in X_keys:
        if key not in _data:
            if verbose:
                print(f"Warning: '{key}' not found in data. Skipping.")
                print()
            continue
        
        missing = [col for col in keep if col not in _data[key].columns]
        if missing:
            raise ValueError(
                f"Features not found in data['{key}']: {missing}"
            )
        
        drop = [col for col in _data[key].columns if col not in keep]
        _data[key] = _data[key].drop(columns=drop)

        if verbose:
            cols = _data[key].columns
            print(f"Kept {len(cols)} columns in data['{key}'].")
            if list_features:
                for col in cols:
                    print(f'- {col}')
            print()

    return _data


def clean_data(
    df: pd.DataFrame,
    missing: bool = True,
    infinite: bool = True,
    negative: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop rows with missing, infinite, or negative values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    missing : bool, default True
        Whether to remove rows with missing values
    infinite : bool, default True
        Whether to remove rows with np.inf or -np.inf values
    negative : bool, default True
        Whether to remove rows with negative values
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with rows removed
    """
    if verbose:
        print('='*70)
        print('Clean Data')
        print('-'*70)
    
    df = df.copy()
    n_initial = len(df)

    if missing:
        n_before_missing = len(df)
        df.dropna(inplace=True)
        n_after_missing = len(df)
        n_rmv_missing = n_before_missing - n_after_missing

        if verbose:
            print(f'Dropped {n_rmv_missing:,} rows with missing values.')

    if infinite or negative:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

    if infinite:
        n_before_infinite = len(df)
        mask = ~(df[numeric_cols].isin([np.inf, -np.inf])).any(axis=1)
        df = df[mask]
        n_after_infinite = len(df)
        n_rmv_infinite = n_before_infinite - n_after_infinite

        if verbose:
            print(f'Dropped {n_rmv_infinite:,} rows with infinite values.')

    if negative:
        n_before_negative = len(df)
        df = df[(df[numeric_cols] >= 0).all(axis=1)]
        n_after_negative = len(df)
        n_rmv_negative = n_before_negative - n_after_negative

        if verbose:
            print(f'Dropped {n_rmv_negative:,} rows with negative values.')

    df.reset_index(drop=True, inplace=True)

    if verbose:
        n_final = len(df)
        print()
        print('-'*70)
        print(f'Rows Before: {n_initial:,}')
        print(f'Rows After:  {n_final:,}')
        print()

    return df


def prepare_labels(
    df: pd.DataFrame,
    raw_col: str = 'label',
    processed_col: str = 'label',
    drop_labels: list = None,
    replace_labels: dict = None,
    clean_labels: bool = True,
    drop_raw: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare labels for machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    raw_col : str, default 'label'
        Name of column containing raw labels
        The function will convert this to '_raw_label' to allow for
        comparison with processed labels in 'label' column. 
        If drop_raw=True, '_raw_label' will be dropped.
    processed_col : str, default 'label'
        Name of column to add with processed labels
    drop_labels : list, default None
        List of label values to remove
    replace_labels : dict, default None
        Mapping of old label values (keys) to new label values
        This is passed to df.replace().
    clean_labels : bool, default True
        Whether to clean label values
        This will convert label values to lowercase, remove outside
        whitespace, and replace inside whitespace and special characters
        with underscores.
    drop_raw : bool, default True
        Whether to drop the column containing raw labels
    verbose : bool, default True
        Print information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with prepared labels
    """
    if verbose:
        print('='*70)
        print('Prepare Labels')
        print('-'*70)
    
    df = df.copy()
    n_initial = len(df)
    
    df['_raw_label'] = df[raw_col]
    df[processed_col] = df['_raw_label']

    if drop_labels:
        n_before_drop = len(df)
        df = df[~df[processed_col].isin(drop_labels)]
        n_after_drop = len(df)
        n_rmv_drop = n_before_drop - n_after_drop

        if verbose:
            print(f'Dropped {n_rmv_drop:,} rows with labels:')
            for label in drop_labels:
                print(f'- {label}')
            print()

        if len(df) == 0:
            warnings.warn(
                "All rows were dropped. Returning empty DataFrame.",
                UserWarning
            )

    if replace_labels:
        df[processed_col] = df[processed_col].replace(replace_labels)

        if verbose:
            print('Replaced label values:')
            for k, v in replace_labels.items():
                print(f'- {k} -> {v}')
            print()

    if clean_labels:
        def clean_label(label):
            cleaned = str(label).lower().strip()
            cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
            return cleaned.strip('_')
        
        df[processed_col] = df[processed_col].apply(clean_label)

        if verbose:
            print('Cleaned label values.')
            print()

    df.reset_index(drop=True, inplace=True)

    if verbose:
        print('-'*70)
        print('Label Distribution:')
        if len(df) > 0:
            print(df.groupby('_raw_label')[processed_col]
                .value_counts()
                .sort_index()
                .to_string())
        else:
            print('(empty)')
        print()

    if drop_raw:
        df.drop(columns='_raw_label', inplace=True)

    if verbose:
        n_final = len(df)
        print('-'*70)
        print(f'Rows Before: {n_initial:,}')
        print(f'Rows After:  {n_final:,}')
        print()

    return df
