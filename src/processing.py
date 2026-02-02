"""
Process data in machine learning training pipeline.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        df = df[(df >= 0).all(axis=1)]
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


def prepare_labels_binary(
    df: pd.DataFrame,
    label_col: str = 'label',
    benign_value: str = 'BENIGN',
    exclude_values: Optional[List[str]] = None,
    drop_original: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create binary attack labels from multi-class labels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with original labels
    label_col : str, default 'Label'
        Name of column containing original labels
    benign_value : str, default 'BENIGN'
        Label value indicating benign/normal traffic
    exclude_values : list of str, optional
        Label values to exclude from dataset
    drop_original : bool, default True
        Whether to drop the original label column
    verbose : bool, default True
        Print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_attack' binary label column
    """
    if verbose:
        print("="*70)
        print("Prepare Labels")
        print("-"*70)
    
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame")
    
    df = df.copy()

    # Exclude Values
    if exclude_values is not None:
        initial_rows = len(df)
        df = df[~df[label_col].isin(exclude_values)]
        rows_rm = initial_rows - len(df)
        
        if verbose:
            rows_rm_p = rows_rm/initial_rows*100
            print(f"Values Removed:")
            print(f"Removed {rows_rm:,} ({rows_rm_p:.2f}%) rows with labels:")
            print(f"{exclude_values}")
            print()
    
    # Create Labels
    df['is_attack'] = (df[label_col] != benign_value).astype(int)
    
    if verbose:
        print(f"Labels Created:")
        print(df['is_attack'].value_counts().to_dict())
        print()

        print(f"Original Labels by Created Labels:")
        print(df.groupby('is_attack')[label_col].value_counts())
        print()
    
    # Drop Original Labels
    if drop_original:
        df = df.drop(label_col, axis=1)
    
    return df


def split_data(
    df: pd.DataFrame, 
    target_col: str = 'is_attack', 
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
    target_col : str, default 'is_attack'
        Name of target column for stratification
    test_size : float, default 0.2
        Proportion of dataset to include in test split
    stratify_y : bool, default True
        Whether to stratify split by target_col
    random_state : int, default 76
        Random state for train_test_split()
    verbose : bool, default True
        Print information including class balance comparison
        
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
    
    # Split Data
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
        # Full
        fll_cnts = y.value_counts().sort_index()
        fll_pcts = (fll_cnts / len(y) * 100).round(1)
        
        # Train
        trn_cnts = y_train.value_counts().sort_index()
        trn_pcts = (trn_cnts / len(y_train) * 100).round(1)
        
        # Test
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

        print(f"Class Balance Comparison:")
        print("-"*70)
        print(f"{'Class':<10} {'Full Dataset':<20} {'Training Set':<20} "
              f"{'Test Set':<20}")
        print("-"*70)
        
        for class_val in fll_cnts.index:
            fll_str = (f"{fll_cnts[class_val]:>6,} "
                       f"({fll_pcts[class_val]:>4.1f}%)")
            trn_str = (f"{trn_cnts[class_val]:>6,} "
                       f"({trn_pcts[class_val]:>4.1f}%)")
            tst_str = (f"{tst_cnts[class_val]:>6,} "
                       f"({tst_pcts[class_val]:>4.1f}%)")
            
            class_name = "Benign" if class_val == 0 else "Attack"
            print(f"{class_name:<10} {fll_str:<20} {trn_str:<20} "
                  f"{tst_str:<20}")
        
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
    
    # Return as dictionary
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
