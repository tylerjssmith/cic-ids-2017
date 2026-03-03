"""Process data for network intrusion detection."""
import copy
import pandas as pd 


def drop_features(
    df: pd.DataFrame, 
    drop: list[str], 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Drop features by name from DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop : list, default None
        List of column names to drop
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with named features removed
    """    
    if verbose:
        print('='*70)
        print('Drop Features by Name')
        print('-'*70)
        n_cols_before = len(df.columns)

    not_found = [col for col in drop if col not in df.columns]
    if not_found:
        raise ValueError(f'{not_found} not found in df')

    df = df.drop(columns=drop)

    if verbose:
        n_cols_dropped = len(drop)
        print(f'Dropped {n_cols_dropped} columns:')
        for col in drop:
            print(f'- {col}')
        print()

        n_cols_after = len(df.columns)
        print('-'*70)
        print(f'Columns Before: {n_cols_before}')
        print(f'Columns After:  {n_cols_after}')
        print()

    return df


def indicate_service(
    df: pd.DataFrame, 
    service_port_map: dict[str, list[int]],
    port_column: str = 'destination_port',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Indicate service name prior to dropping port-number column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    service_port_map : dict[str, list[int]]
        Mapping of service names to lists of port numbers
        Example: {'ssh': [22], 'ftp': [20,21]}
    port_column : str, default 'destination_port'
        Name of column with port numbers to drop
    verbose : bool, default True
        Print information

    Returns
    -------
    pd.DataFrame
        DataFrame with indicator variables for services instead of
        column with port numbers.
    """
    if verbose:
        print('='*70)
        print('Indicate Services')
        print('-'*70)

    df = df.copy()

    for service, ports in service_port_map.items():
        service_column = f'is_{service}'
        df[service_column] = df[port_column].isin(ports).astype(int)

        if verbose:
            print(f'Ports {ports} -> {service_column}')

    df = df.drop(columns=port_column)

    if verbose:
        print()
        print(f'{port_column} was dropped.')
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
