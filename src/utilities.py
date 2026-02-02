"""
Load and save data in machine learning training pipeline.
"""
import joblib
from pathlib import Path
from typing import Literal
import pandas as pd


def load_data(
    filepath: Path,
    data_type: Literal['csv', 'parquet'] = 'csv',
    clean_columns: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load data from file with optional column cleaning.
    
    Parameters
    ----------
    filepath : Path
        Path to data file
    data_type : {'csv', 'parquet'}, default 'csv'
        Type of data file to load
    clean_columns : bool, default True
        Clean column names (strip, lowercase, replace spaces)
    verbose : bool, default True
        Print information
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    if verbose:
        print("="*70)
        print("Load Data")
        print("-"*70)
        
    if data_type.lower() == 'csv':
        df = pd.read_csv(filepath)
    elif data_type.lower() == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(
            f"data_type must be 'csv' or 'parquet', not {data_type}"
        )
    
    if clean_columns:
        df.columns = df.columns\
            .str.strip()\
            .str.lower()\
            .str.replace(' ', '_')\
            .str.replace('/', '_')
    
    if verbose:
        print(f"Loaded:  {filepath}")
        print(f"Rows:    {len(df):,}")
        print(f"Columns: {len(df.columns):,}")
        print()
    
    return df


def save_data_splits(
    data: dict, 
    filepath: str, 
    verbose: bool = True
):
    """
    Save data to file.

    Parameters
    ----------
    data : dict
        Data to write
    filepath : str
        Where to write data
    verbose: bool, default True
        Print information
    """
    if verbose:
        print("="*70)
        print("Save Data Splits")
        print("-"*70)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, filepath, compress=("gzip",3))

    if verbose:
        print(f"Saved: {filepath}")
        print()

    return data