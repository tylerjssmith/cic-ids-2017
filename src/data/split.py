"""Split data for training network intrusion detection models."""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame, 
    label_col: str = 'label', 
    test_size: float = 0.2, 
    random_state: int = 76, 
    stratify: bool = True, 
    verbose: bool = True
) -> dict:
    """
    Split data for training and testing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    label_col : str, default 'label'
        Label column
    test_size : float, default 0.2
        Proportion of data for test set
    random_state : int, default 76
        Random state for train_test_split()
    stratify : bool, default True
        Whether to stratify on label_col
    verbose : bool, default True
        Print information

    Returns
    -------
    dict
        Dictionary containing:
        - X_train (pd.DataFrame)
        - X_test (pd.DataFrame)
        - y_train (pd.Series)
        - y_test (pd.Series)
    """
    if verbose:
        print('='*70)
        print('Split Data')
        print('-'*70)

    if label_col not in df.columns:
        raise ValueError(f'Column "{label_col}" not found in df')

    X = df.drop(columns=[label_col])
    y = df[label_col]

    if verbose:
        print(f'Test Size:    {test_size}')
        print(f'Random State: {random_state}')
        print(f'Stratify:     {stratify}')
        print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    if verbose:
        _print_split_sizes(df, X_train, X_test)
        _print_class_balance(y, y_train, y_test, stratify)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def _print_split_sizes(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> None:
    """
    Print dataset sizes for full, training, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    """
    train_pct = len(X_train) / len(df) * 100
    test_pct = len(X_test) / len(df) * 100

    print('Dataset Sizes:')
    print(f'Full:     {len(df):>10,} rows')
    print(f'Training: {len(X_train):>10,} rows ({train_pct:.1f}%)')
    print(f'Test:     {len(X_test):>10,} rows ({test_pct:.1f}%)')
    print()


def _print_class_balance(
    y: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    stratify: bool
) -> None:
    """
    Print class balance comparison table and stratification check.

    Parameters
    ----------
    y : pd.Series
        Full label series
    y_train : pd.Series
        Training label series
    y_test : pd.Series
        Test label series
    stratify : bool
        Whether stratification was applied
    """
    full_cnts = y.value_counts().sort_index()
    full_pcts = (full_cnts / len(y) * 100).round(1)

    train_cnts = y_train.value_counts().reindex(full_cnts.index, fill_value=0)
    train_pcts = (train_cnts / len(y_train) * 100).round(1)

    test_cnts = y_test.value_counts().reindex(full_cnts.index, fill_value=0)
    test_pcts = (test_cnts / len(y_test) * 100).round(1)

    print('Class Balance Comparison:')
    print('-'*70)
    print(f'{"Class":<18} '
          f'{"Full Dataset":<18} '
          f'{"Training Set":<18} '
          f'{"Test Set":<16}')
    print('-'*70)

    max_count_width = max(
            len(f'{full_cnts.max():,}'),
            len(f'{train_cnts.max():,}'),
            len(f'{test_cnts.max():,}')
        )

    for class_val in full_cnts.index:
            full_str = (
                f'{full_cnts[class_val]:>{max_count_width},} '
                f'({full_pcts[class_val]:>4.1f}%)'
            )
            train_str = (
                f'{train_cnts[class_val]:>{max_count_width},} '
                f'({train_pcts[class_val]:>4.1f}%)'
            )
            test_str = (
                f'{test_cnts[class_val]:>{max_count_width},} '
                f'({test_pcts[class_val]:>4.1f}%)'
            )
            print(
                f'{str(class_val):<16} '
                f'{full_str:>16} '
                f'{train_str:>16} '
                f'{test_str:>16}'
            )

    print('-'*70)

    if stratify:
        max_diff = max(
            abs(train_pcts - full_pcts).max(),
            abs(test_pcts - full_pcts).max()
        )
        if max_diff < 0.5:
            print('Success: Class distribution differences <0.5%')
        else:
            print(f'Class distribution difference: {max_diff:.2f}%')
    else:
        print(f'Stratification disabled (stratify={stratify})')

    print()