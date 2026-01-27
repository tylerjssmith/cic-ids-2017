import yaml
from pathlib import Path
from typing import Literal
import pandas as pd

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """
    Load configuration file.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration values
    
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_path(stage: str, filename: str) -> Path:
    """
    Use configuration file to make file path to data.

    Parameters
    ----------
    stage : str
        Stage of data processing (raw, intermediate, processed) used
        to select directory from configuration file
    filename : str
        Filename to read or write

    Returns
    -------
    Path
        File path to read or write
    """
    config = load_config()['data']['dir']
    if stage not in config:
        raise ValueError(f"stage must be one of: {list(config.keys())}")
    dirname = Path(config[stage])
    return dirname / filename


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
        print(f"Loaded {len(df):,} rows from: {filename}\n")
    
    return df