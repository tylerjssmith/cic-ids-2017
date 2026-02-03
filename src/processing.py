"""
Process data in machine learning training pipeline.
"""
import re
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_labels(
    df: pd.DataFrame, 
    label_col: str = 'label',
    exclude_values: list = None,
    replace_values: dict = None, 
    clean_values: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare labels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with original labels
    label_col : str
        Name of label column to process
    exclude_values : list, optional
        List of label values to exclude (rows will be dropped)
    replace_values : dict, optional
        Dictionary mapping new values (keys) to old values to replace
        Values can be strings or lists of strings
    clean_values : bool, default True
        lowercase, strip whitespace, replace inner whitespace and 
        non-alphanumeric chars with underscores
    verbose : bool, default True
        Print information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed labels
    """
    if verbose:
        print("="*70)
        print("Prepare Labels")
        print("-"*70)

    df = df.copy()
    df.rename(columns={label_col: 'original'}, inplace=True)
    df['label'] = df['original']

    if verbose:
        initial_rows = len(df)
        print(f"Initial Rows: {initial_rows:,}")
        print()
        print("Initial Distribution:")
        print(df['label'].value_counts().sort_index().to_string())
        print()
    
    if exclude_values is not None and len(exclude_values) > 0:
        rows_rm = df['label'].isin(exclude_values).sum()
        df = df[~df['label'].isin(exclude_values)]

        if verbose:
            print(f"Removed {rows_rm:,} rows with labels:")
            for v in exclude_values:
                print(f"- {v}")
            print()
    
    if replace_values is not None and len(replace_values) > 0:
        reverse_map = {}
        for new_value, old_values in replace_values.items():
            if isinstance(old_values, list):
                for old_value in old_values:
                    reverse_map[old_value] = new_value
            else:
                reverse_map[old_values] = new_value
        
        df['label'] = df['label'].replace(reverse_map)
        
        if verbose:
            print("Mapped old to new values:")
            for new_val, old_vals in replace_values.items():
                for old_val in list(old_vals):
                    print(f"- {old_val:<19} -> {new_val:<}")
            print()
    
    if clean_values:
        def clean_label(label):
            cleaned = str(label).lower().strip()
            cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
            return cleaned.strip('_')
        
        df['label'] = df['label'].apply(clean_label)
    
    if verbose:
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        perc_removed = rows_removed / initial_rows * 100

        print('-'*70)
        print(f"Final Rows: {final_rows:,}")
        print(f"Total Removed: {rows_removed:,} ({perc_removed:.4f}%)")
        print()
        print("Final Distribution:")
        print(df.groupby('original')['label'].value_counts().map('{:,}'.format).to_string())
        print()
    
    df.drop(columns=['original'], inplace=True)
    
    return df


def drop_features(
    df: pd.DataFrame, 
    drop: list = [], 
    rm_zv: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop features
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop : list
        Drop columns
    rm_zv : bool, default True
        Drop zero-variance columns
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with features dropped
    """
    if verbose:
        print('='*70)
        print('Drop Features')
        print('-'*70)
        print(f'Initial Columns: {len(df.columns)}')
        print()

    df = df.copy()

    if len(drop) > 0:
        df = df.drop(columns=list(drop))

        if verbose:
            print('Dropped named columns if they exist:')
            for v in drop:
                print(f'- {v}')
            print()

    if rm_zv:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zv = df[numeric_cols].var()[df[numeric_cols].var() == 0].index.tolist()
        df = df.drop(columns=list(zv))

        if verbose and len(zv) > 0:
            print('Dropped zero-variance columns:')
            for v in zv:
                print(f'- {v}')
            print()

    if verbose:
        print('-'*70)
        print(f"Final Columns: {len(df.columns)}")
        print(f"Dropped {len(drop)} columns")
        print()

    return df


def clean_data(
    df: pd.DataFrame,
    rm_nan: bool = True,
    rm_inf: bool = True, 
    rm_neg: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Clean DataFrame by removing missing values and infinities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    rm_nan : bool, default True
        Remove rows with NaN values
    rm_inf : bool, default True
        Remove rows with np.inf or -np.inf values
    rm_neg : bool, default True
        Remove rows with values <0
    verbose : bool, default True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    if verbose:
        print("="*70)
        print("Clean Data")
        print("-"*70)

    df = df.copy()
    initial_rows = len(df)

    if verbose:
        print(f"Initial Rows: {len(df):,}")
        print()
    
    # Remove NaN
    if rm_nan:
        rows_before_nan = len(df)
        df.dropna(inplace=True)
        rows_after_nan = len(df)
        nan_rm = rows_before_nan - rows_after_nan
        
        if verbose and nan_rm > 0:
            nan_rm_p = nan_rm/initial_rows*100
            print(
                f"Removed {nan_rm:,} rows with NaN values ({nan_rm_p:.2f}%)"
            )
    
    # Remove np.inf and -np.inf
    if rm_inf:
        rows_before_inf = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        rows_after_inf = len(df)
        inf_rm = rows_before_inf - rows_after_inf

        if verbose and inf_rm > 0:
            inf_rm_p = inf_rm/initial_rows*100
            print(
                f"Removed {inf_rm:,} rows with values <0 ({inf_rm_p:.2f}%)"
            )

    # Remove negative
    if rm_neg:
        rows_before_neg = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[(df[numeric_cols] >= 0).all(axis=1)]
        rows_after_neg = len(df)
        neg_rm = rows_before_neg - rows_after_neg

        if verbose and neg_rm > 0:
            neg_rm_p = neg_rm/initial_rows*100
            print(
                f"Removed {neg_rm:,} rows with negative values ({neg_rm_p:.2f}%)"
            )

    # Summary    
    if verbose:
        total_rm = initial_rows - len(df)
        total_rm_p = total_rm/initial_rows*100
        print(f"\n{'-'*70}")
        print(f"Final Rows: {len(df):,}")
        print(f"Total Removed: {total_rm:,} ({total_rm_p:.2f}%)")
        print()
    
    return df


def split_data(
    df: pd.DataFrame, 
    target_col: str = 'label', 
    test_size: float = 0.2,
    stratify_y: bool = True,
    random_state: int = 76,
    verbose: bool = True
):
    """
    Split DataFrame into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, default 'label'
        Name of label column
    test_size : float, default 0.2
        Proportion of dataset to include in test split
    stratify_y : bool, default True
        Whether to stratify split by target_col
    random_state : int, default 76
        Random state for train_test_split()
    verbose : bool, default True
        Print information
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train': Training features (DataFrame)
        - 'X_test': Test features (DataFrame)
        - 'y_train': Training labels (Series)
        - 'y_test': Test labels (Series)
    """
    if verbose:
        print("="*70)
        print("Split Data")
        print("-"*70)

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify_y else None
    )
    
    if verbose:
        # Summary
        fll_cnts = y.value_counts().sort_index()
        fll_pcts = (fll_cnts / len(y) * 100).round(1)
        
        trn_cnts = y_train.value_counts().sort_index()
        trn_pcts = (trn_cnts / len(y_train) * 100).round(1)
        
        tst_cnts = y_test.value_counts().sort_index()
        tst_pcts = (tst_cnts / len(y_test) * 100).round(1)
        
        # Comparison
        print(f"Dataset Sizes:")
        print(f"Full:     {len(df):>8,} rows")
        trn_pcts_size = len(X_train)/len(df)*100
        print(f"Training: {len(X_train):>8,} rows ({trn_pcts_size:.1f}%)")
        tst_pcts_size = len(X_test)/len(df)*100
        print(f"Test:     {len(X_test):>8,} rows ({tst_pcts_size:.1f}%)")
        print()

        max_class_len = max(len(str(class_val)) for class_val in fll_cnts.index)
        class_col_width = max(max_class_len, 5)
        
        print(f"Class Balance Comparison:")
        print("-"*70)
        print(f"{'Class':<{class_col_width}} {'Full Dataset':>18} "
              f"{'Training Set':>16} {'Test Set':>16}")
        print("-"*70)
        
        for class_val in fll_cnts.index:
            fll_str = (f"{fll_cnts[class_val]:>6,} "
                       f"({fll_pcts[class_val]:>4.1f}%)")
            trn_str = (f"{trn_cnts[class_val]:>6,} "
                       f"({trn_pcts[class_val]:>4.1f}%)")
            tst_str = (f"{tst_cnts[class_val]:>6,} "
                       f"({tst_pcts[class_val]:>4.1f}%)")
            
            print(f"{str(class_val):<{class_col_width}} {fll_str:>18} "
                  f"{trn_str:>16} {tst_str:>16}")
        
        print("-"*70)
        
        # Stratification
        if stratify_y:
            max_diff = max(abs(trn_pcts - fll_pcts).max(),
                           abs(tst_pcts - fll_pcts).max())
            if max_diff < 0.5:
                print("Stratification successful "
                      "(class distribution differences <0.5%)")
            else:
                print(f"Class distribution difference: {max_diff:.2f}%")
        else:
            print("Note: Stratification disabled")
        
        print()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }