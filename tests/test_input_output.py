"""Tests for input_output.py."""
import pytest
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from input_output import (
    load_data, 
    save_data_splits,
    load_data_splits, 
    load_results
)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        ' Flow Duration': [1, 2, 3],
        'Total/Fwd Packets': [10, 20, 30],
        'Label': ['BENIGN', 'DoS', 'BENIGN'],
    })


@pytest.fixture
def csv_dir(tmp_path, sample_df):
    """Directory with two CSV files."""
    sample_df.to_csv(tmp_path / 'Monday.csv', index=False)
    sample_df.to_csv(tmp_path / 'Tuesday.csv', index=False)
    return tmp_path


@pytest.fixture
def splits(sample_df):
    X = sample_df.drop(columns=['Label'])
    y = sample_df['Label']
    return {
        'X_train': X.iloc[:2].reset_index(drop=True),
        'X_test':  X.iloc[2:].reset_index(drop=True),
        'y_train': y.iloc[:2].reset_index(drop=True),
        'y_test':  y.iloc[2:].reset_index(drop=True),
    }


# --- load_data ---------------------------------------------------------------
def test_concatenates_all_csvs_and_returns_dataframe(csv_dir, sample_df):
    """All CSV files in the directory are loaded and concatenated."""
    df = load_data(csv_dir, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_df) * 2


def test_clean_names_strips_and_lowercases_columns(csv_dir):
    """Column names are stripped, lowercased, and special chars replaced."""
    df = load_data(csv_dir, clean_names=True, verbose=False)
    for col in df.columns:
        assert col == col.strip()
        assert col == col.lower()
        assert ' ' not in col
        assert '/' not in col


# --- save_data_splits --------------------------------------------------------
def test_creates_parquet_files_for_all_splits(tmp_path, splits):
    """A parquet file is written for every key in the data dict."""
    save_data_splits(splits, tmp_path, prefix='', verbose=False)
    for key in splits:
        assert (tmp_path / f'{key}.parquet').exists()


def test_raises_on_non_dataframe_or_series_value(tmp_path):
    """TypeError is raised when a dict value is not a DataFrame or Series."""
    bad_data = {'X_train': np.array([1, 2, 3])}
    with pytest.raises(TypeError):
        save_data_splits(bad_data, tmp_path, verbose=False)


# --- load_data_splits --------------------------------------------------------
def test_returns_correct_keys_and_types(tmp_path, splits):
    """Returned dict has expected keys; y splits are pd.Series."""
    save_data_splits(splits, tmp_path, prefix='', verbose=False)
    loaded = load_data_splits(
        tmp_path,
        X_train='X_train.parquet',
        X_test='X_test.parquet',
        y_train='y_train.parquet',
        y_test='y_test.parquet',
        verbose=False,
    )
    assert set(loaded.keys()) == {'X_train', 'X_test', 'y_train', 'y_test'}
    assert isinstance(loaded['y_train'], pd.Series)
    assert isinstance(loaded['y_test'],  pd.Series)


def test_raises_on_missing_file_data(tmp_path, splits):
    """FileNotFoundError is raised when a required parquet file is absent."""
    splits['X_train'].to_parquet(tmp_path / 'X_train.parquet')
    with pytest.raises(FileNotFoundError):
        load_data_splits(tmp_path, verbose=False)


# --- load_results ------------------------------------------------------------
def test_loads_joblib_artifact_correctly(tmp_path):
    """Saved joblib artifact is returned intact."""
    payload = {'accuracy': 0.99, 'model': 'RandomForest'}
    joblib.dump(payload, tmp_path / 'results.pkl')
    loaded = load_results(tmp_path, 'results.pkl', verbose=False)
    assert loaded == payload


def test_raises_on_missing_file_results(tmp_path):
    """FileNotFoundError is raised when the results file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path, 'nonexistent.pkl', verbose=False)
