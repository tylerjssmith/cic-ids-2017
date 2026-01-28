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
                f"Removed {inf_rm:,} rows with np.inf values ({inf_rm_p:.2f}%)"
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
        full_counts = y.value_counts().sort_index()
        full_pct = (full_counts / len(y) * 100).round(1)
        
        # Train
        train_counts = y_train.value_counts().sort_index()
        train_pct = (train_counts / len(y_train) * 100).round(1)
        
        # Test
        test_counts = y_test.value_counts().sort_index()
        test_pct = (test_counts / len(y_test) * 100).round(1)
        
        # Comparison
        print(f"Dataset Sizes:")
        print(f"Full:     {len(df):>8,} rows")
        print(f"Training: {len(X_train):>8,} rows ({len(X_train)/len(df)*100:.1f}%)")
        print(f"Test:     {len(X_test):>8,} rows ({len(X_test)/len(df)*100:.1f}%)")
        print()

        print(f"Class Balance Comparison:")
        print("-"*70)
        print(f"{'Class':<10} {'Full Dataset':<20} {'Training Set':<20} {'Test Set':<20}")
        print("-"*70)
        
        for class_val in full_counts.index:
            full_str = f"{full_counts[class_val]:>6,} ({full_pct[class_val]:>4.1f}%)"
            train_str = f"{train_counts[class_val]:>6,} ({train_pct[class_val]:>4.1f}%)"
            test_str = f"{test_counts[class_val]:>6,} ({test_pct[class_val]:>4.1f}%)"
            
            class_name = "Benign" if class_val == 0 else "Attack"
            print(f"{class_name:<10} {full_str:<20} {train_str:<20} {test_str:<20}")
        
        print("-"*70)
        
        # Stratification
        if stratify_y:
            max_diff = max(abs(train_pct - full_pct).max(), abs(test_pct - full_pct).max())
            if max_diff < 0.5:
                print("Stratification successful (class distribution differences <0.5%)")
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
