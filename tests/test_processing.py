"""Tests for processing.py."""
import numpy as np
import pandas as pd
import pytest

from processing import (
    drop_features, 
    keep_features, 
    clean_data, 
    prepare_labels
)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'flow_duration':    [1, 2, 3, 4],
        'total_fwd_packets': [10, 20, 30, 40],
        'constant':         [7, 7, 7, 7],   # zero-variance
        'label':            ['BENIGN', 'DoS', 'BENIGN', 'DoS'],
    })


@pytest.fixture
def splits(sample_df):
    X = sample_df.drop(columns=['label'])
    return {
        'X_train': X.iloc[:3].reset_index(drop=True),
        'X_test':  X.iloc[3:].reset_index(drop=True),
    }


# --- drop_features -----------------------------------------------------------
def test_drop_features_removes_named_columns(sample_df):
    """Columns listed in drop are removed from the result."""
    result = drop_features(sample_df, drop=['label'], zero_variance=False, 
                           verbose=False)
    assert 'label' not in result.columns
    assert len(result.columns) == len(sample_df.columns) - 1


def test_drop_features_removes_zero_variance_columns(sample_df):
    """Columns with zero variance are automatically removed."""
    result = drop_features(sample_df, drop=None, zero_variance=True, 
                           verbose=False)
    assert 'constant' not in result.columns


#  --- keep_features ----------------------------------------------------------
def test_keep_features_retains_only_specified_columns(splits):
    """Only the listed columns remain in each X split."""
    result = keep_features(splits, keep=['flow_duration'], verbose=False)
    assert list(result['X_train'].columns) == ['flow_duration']
    assert list(result['X_test'].columns)  == ['flow_duration']


def test_keep_features_raises_on_missing_column(splits):
    """ValueError is raised when a requested column does not exist."""
    with pytest.raises(ValueError):
        keep_features(splits, keep=['nonexistent_feature'], verbose=False)


# --- clean_data --------------------------------------------------------------
def test_clean_data_removes_rows_with_problem_values():
    """Rows containing NaN, infinite, and negative values are all dropped."""
    df = pd.DataFrame({
        'a': [1.0,  np.nan, np.inf, -1.0],
        'b': [1.0,  2.0,    3.0,    4.0],
    })
    result = clean_data(df, verbose=False)
    assert len(result) == 1
    assert result.iloc[0]['a'] == 1.0


def test_clean_data_resets_index(sample_df):
    """Returned DataFrame has a clean 0-based RangeIndex."""
    df = pd.DataFrame({
        'a': [1.0, np.nan, 3.0],
        'b': [1.0, 2.0,    3.0],
    })
    result = clean_data(df, verbose=False)
    assert list(result.index) == list(range(len(result)))


# --- prepare_labels ----------------------------------------------------------
def test_prepare_labels_cleans_label_values():
    """Labels are lowercased and special characters replaced with underscores."""
    df = pd.DataFrame({'label': ['BENIGN', 'DoS Hulk', 'Web Attack – Brute Force']})
    result = prepare_labels(df, clean_labels=True, verbose=False)
    assert list(result['label']) == ['benign', 'dos_hulk', 'web_attack_brute_force']


def test_prepare_labels_drops_specified_labels():
    """Rows whose label appears in drop_labels are removed."""
    df = pd.DataFrame({'label': ['BENIGN', 'Heartbleed', 'DoS', 'Heartbleed']})
    result = prepare_labels(df, drop_labels=['Heartbleed'], clean_labels=False, 
                            verbose=False)
    assert 'heartbleed' not in result['label'].str.lower().values
    assert len(result) == 2
