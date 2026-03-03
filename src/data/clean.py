"""Clean data for network intrusion detection."""
import numpy as np
import pandas as pd


def drop_zero_variance(
    df: pd.DataFrame, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop features with zero variance from DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with zero-variance features removed
    """
    if verbose:
        print('='*70)
        print('Drop Features with Zero Variance')
        print('-'*70)
        n_cols_before = len(df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variance = (
        df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .var(skipna=True)
    )
    zv = variance[variance == 0].index.tolist()
    df = df.drop(columns=zv)

    if verbose:
        n_cols_dropped = len(zv)
        print(f'Dropped {n_cols_dropped} columns with zero variance:')
        for col in zv:
            print(f'- {col}')
        print()

        n_cols_after = len(df.columns)
        print('-'*70)
        print(f'Columns Before: {n_cols_before}')
        print(f'Columns After:  {n_cols_after}')
        print()

    return df


def remove_missing(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove rows with missing values

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with rows with missing values removed
    """
    if verbose:
        print('='*70)
        print('Remove Rows with Missing Values')
        print('-'*70)
        n_before = len(df)
    
    df = df.dropna().reset_index(drop=True)

    if verbose:
        n_after = len(df)
        n_removed = n_before - n_after
        print(f'Dropped {n_removed:,} rows with missing values.')
        print()

    return df


def remove_infinite(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove rows with infinite values

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with rows with infinite values removed
    """
    if verbose:
        print('='*70)
        print('Remove Rows with Infinite Values')
        print('-'*70)    
        n_before = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = ~np.isinf(df[numeric_cols]).any(axis=1)
    df = df[mask]
    df = df.reset_index(drop=True)

    if verbose:
        n_after = len(df)
        n_removed = n_before - n_after
        print(f'Dropped {n_removed:,} rows with infinite values.')
        print()

    return df


def remove_negative(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove rows with negative values

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with rows with negative values removed
    """
    if verbose:
        print('='*70)
        print('Remove Rows with Negative Values')
        print('-'*70)    
        n_before = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(df[numeric_cols] >= 0).all(axis=1)]
    df = df.reset_index(drop=True)

    if verbose:
        n_after = len(df)
        n_removed = n_before - n_after
        print(f'Dropped {n_removed:,} rows with negative values.')
        print()

    return df
