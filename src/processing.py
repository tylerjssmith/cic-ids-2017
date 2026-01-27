"""
Process and split data for network intrusion detection system.
"""

import numpy as np
import pandas as pd
from typing import List, Literal, Optional
from sklearn.model_selection import train_test_split
from src.utils import load_config, make_path


def load_data(
    stage: str,
    filename: str,
    data_type: Literal['csv', 'parquet'] = 'csv',
    clean_columns: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load data from file with optional column cleaning.
    
    Parameters
    ----------
    stage : str
        Stage of data processing (raw, intermediate, processed) passed
        to make_path().
    filename : str
        Filename to read
    data_type : {'csv', 'parquet'}, default 'csv'
        Type of data file to load
    clean_columns : bool, default True
        Clean column names (strip, lowercase, replace spaces)
    verbose : bool, default True
        Print loading information
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    if verbose:
        print("="*70)
        print("Load Data")
        print("-"*70)
    
    filepath = make_path(stage, filename)
    
    if data_type.lower() == 'csv':
        df = pd.read_csv(filepath)
    elif data_type.lower() == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported data_type: '{data_type}'. Must be 'csv' or 'parquet'")
    
    if clean_columns:
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if verbose:
        print(f"Loaded: {filename}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns):,}")
        print("="*70)
        print("\n")
    
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean DataFrame by removing missing values and infinities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool, default True
        Print information about rows excluded
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    if verbose:
        print("="*70)
        print("Clean Data")
        print("-"*70)
        print(f"Initial rows: {len(df):,}")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Remove NaN
    rows_before_nan = len(df)
    df.dropna(inplace=True)
    rows_after_nan = len(df)
    nan_removed = rows_before_nan - rows_after_nan
    
    if verbose and nan_removed > 0:
        print(f"\nRemoved {nan_removed:,} rows with NaN values ({nan_removed/initial_rows*100:.2f}%)")
    
    # Remove np.inf
    rows_before_inf = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    rows_after_inf = len(df)
    inf_removed = rows_before_inf - rows_after_inf
    
    if verbose and inf_removed > 0:
        print(f"Removed {inf_removed:,} rows with np.inf values ({inf_removed/initial_rows*100:.2f}%)")
    
    # Summary
    total_removed = initial_rows - len(df)
    
    if verbose:
        print(f"\n{'-'*70}")
        print(f"Final rows: {len(df):,}")
        print(f"Total removed: {total_removed:,} ({total_removed/initial_rows*100:.2f}%)")
        print("="*70)
        print("\n")
    
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
        Input DataFrame with labels
    label_col : str, default 'Label'
        Name of column containing original labels
    benign_value : str, default 'BENIGN'
        Label value indicating benign/normal traffic
    exclude_values : list of str, optional
        Label values to exclude from dataset
    drop_original : bool, default True
        Whether to drop the original label column
    verbose : bool, default True
        Print label distribution information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_attack' binary label column
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame")
    
    df = df.copy()
    
    if verbose:
        print("="*70)
        print("Prepare Labels")
        print("-"*70)

    if exclude_values is not None:
        initial_rows = len(df)
        df = df[~df[label_col].isin(exclude_values)]
        
        if verbose:
            rows_removed = initial_rows - len(df)
            print(f"Excluded {rows_removed:,} rows with labels: {exclude_values}")
    
    df['is_attack'] = (df[label_col] != benign_value).astype(int)
    
    if verbose:
        print(f"\nBinary labels:")
        print(df['is_attack'].value_counts().to_dict())
        print(f"\nOriginal labels by binary label:")
        print(df.groupby('is_attack')[label_col].value_counts())
        print("="*70)
        print("\n")
    
    if drop_original:
        df = df.drop(label_col, axis=1)
    
    return df


def split_data(df: pd.DataFrame, target_col: str = 'is_attack', verbose: bool = True):
    """
    Split DataFrame into train and test sets with optional class balance reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, default 'is_attack'
        Name of target column for stratification
    verbose : bool, default True
        Print class balance comparison
        
    Returns
    -------
    df_train, df_test : pd.DataFrame
        Training and testing DataFrames
    """
    config = load_config()
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    y = df[target_col]
    
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y if config['data']['stratify'] else None
    )
    
    if verbose:
        print("="*70)
        print("Split Data")
        print("-"*70)
        
        # Full
        full_counts = df[target_col].value_counts().sort_index()
        full_pct = (full_counts / len(df) * 100).round(2)
        
        # Training
        train_counts = df_train[target_col].value_counts().sort_index()
        train_pct = (train_counts / len(df_train) * 100).round(2)
        
        # Test
        test_counts = df_test[target_col].value_counts().sort_index()
        test_pct = (test_counts / len(df_test) * 100).round(2)
        
        # Comparison
        print(f"\nDataset Sizes:")
        print(f"  Full dataset:  {len(df):>8,} rows")
        print(f"  Training set:  {len(df_train):>8,} rows ({len(df_train)/len(df)*100:.1f}%)")
        print(f"  Test set:      {len(df_test):>8,} rows ({len(df_test)/len(df)*100:.1f}%)")
        
        print(f"\nClass Balance Comparison:")
        print("-"*70)
        print(f"{'Class':<15} {'Full Dataset':<20} {'Training Set':<20} {'Test Set':<20}")
        print("-"*70)
        
        for class_val in full_counts.index:
            full_str = f"{full_counts[class_val]:>6,} ({full_pct[class_val]:>5.2f}%)"
            train_str = f"{train_counts[class_val]:>6,} ({train_pct[class_val]:>5.2f}%)"
            test_str = f"{test_counts[class_val]:>6,} ({test_pct[class_val]:>5.2f}%)"
            
            class_name = "Benign" if class_val == 0 else "Attack"
            print(f"{class_name:<15} {full_str:<20} {train_str:<20} {test_str:<20}")
        
        print("-"*70)
        
        # Stratification
        if config['data']['stratify']:
            max_diff = max(abs(train_pct - full_pct).max(), abs(test_pct - full_pct).max())
            if max_diff < 0.5:
                print("Stratification successful (class distribution differences <0.5%)")
            else:
                print(f"Class distribution difference: {max_diff:.2f}%")
        else:
            print("Note: Stratification disabled")
        
        print("="*70)
        print("\n")
    
    return df_train, df_test
