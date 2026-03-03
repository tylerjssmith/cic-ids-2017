"""Prepare labels for network intrusion detection."""
import re
import warnings
import pandas as pd

_RAW_LABEL_COL = '_raw_label'


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
    Prepare labels for network intrusion detection.

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
    if _RAW_LABEL_COL in df.columns:
        raise ValueError(
            f"DataFrame already contains a '{_RAW_LABEL_COL}' column. "
            "Please rename or drop it before calling prepare_labels."
        )

    if verbose:
        print('='*70)
        print('Prepare Labels')
        print('-'*70)

    df = df.copy()
    n_before = len(df)

    df[_RAW_LABEL_COL] = df[raw_col]
    df[processed_col] = df[_RAW_LABEL_COL]

    if drop_labels:
        # _drop_labels() uses column indexing, which returns DataFrame
        df = _drop_labels(df, processed_col, drop_labels, verbose=verbose)

    if replace_labels:
        _replace_labels(df, processed_col, replace_labels, verbose=verbose)

    if clean_labels:
        _clean_labels(df, processed_col, verbose=verbose)

    df = df.reset_index(drop=True)

    if verbose:
        print('-'*70)
        print('Label Distribution:')
        if len(df) > 0:
            print(df.groupby(_RAW_LABEL_COL)[processed_col]
                .value_counts()
                .sort_index()
                .to_string())
        else:
            print('(empty)')
        print()

    if drop_raw:
        df = df.drop(columns=_RAW_LABEL_COL)

    if verbose:
        n_after = len(df)
        print('-'*70)
        print(f'Rows Before: {n_before:,}')
        print(f'Rows After:  {n_after:,}')
        print()

    return df


def _drop_labels(
    df: pd.DataFrame,
    processed_col: str,
    drop_labels: list,
    verbose: bool
) -> pd.DataFrame:
    """
    Drop rows with named label values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    processed_col : str
        Name of column with processed labels
    drop_labels : list
        List of label values to remove
    verbose : bool
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with rows with named label values dropped
    """
    n_before = len(df)
    df = df[~df[processed_col].isin(drop_labels)]
    n_after = len(df)
    n_dropped = n_before - n_after

    if verbose:
        print(f'Dropped {n_dropped:,} rows with labels:')
        for label in drop_labels:
            print(f'- {label}')
        print()

    if len(df) == 0:
        warnings.warn(
            "All rows were dropped. Returning empty DataFrame.",
            UserWarning
        )

    return df


def _replace_labels(
    df: pd.DataFrame,
    processed_col: str,
    replace_labels: dict,
    verbose: bool
) -> None:
    """
    Replace label values

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    processed_col : str
        Name of column with processed labels
    replace_labels : dict
        Mapping of old label values (keys) to new label values
    verbose : bool
        Print information

    Returns
    -------
    None
    """
    df[processed_col] = df[processed_col].replace(replace_labels)

    if verbose:
        print('Replaced label values:')
        for k, v in replace_labels.items():
            print(f'- {k} -> {v}')
        print()


def _clean_labels(
    df: pd.DataFrame,
    processed_col: str,
    verbose: bool
) -> None:
    """
    Clean label values

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    processed_col : str
        Name of column with processed labels
    verbose : bool
        Print information

    Returns
    -------
    None
    """
    def _clean_label(label: str) -> str:
        cleaned = str(label).lower().strip()
        cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
        return cleaned.strip('_')

    df[processed_col] = df[processed_col].apply(_clean_label)

    if verbose:
        print('Cleaned label values.')
        print()