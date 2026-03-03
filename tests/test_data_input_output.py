"""Tests for data_io.py."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data.input_output import (
    load_data, 
    save_data_splits,
    load_data_splits
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
def test_load_data(csv_dir, sample_df):
    df = load_data(csv_dir, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_df) * 2


def test_load_data_raises_on_directory_not_found():
    with pytest.raises(ValueError):
        load_data('not_directory', verbose=False)


def test_load_data_raises_on_filenames_not_list(csv_dir):
    with pytest.raises(TypeError):
        load_data(csv_dir, filenames='not_list.csv', verbose=False)


def test_load_data_raises_on_file_not_found(csv_dir):
    with pytest.raises(FileNotFoundError):
        load_data(csv_dir, filenames=['not_file.csv'], verbose=False)


def test_load_data_clean_names(csv_dir):
    df = load_data(csv_dir, clean_names=True, verbose=False)
    for col in df.columns:
        assert col == col.strip()
        assert col == col.lower()
        assert ' ' not in col
        assert '/' not in col


# --- save_data_splits --------------------------------------------------------
def test_save_data_splits_creates_parquet_files(tmp_path, splits):
    save_data_splits(splits, tmp_path, prefix='', verbose=False)
    for key in splits:
        assert (tmp_path / f'{key}.parquet').exists()


def test_raises_on_non_dataframe_or_series_value(tmp_path):
    bad_data = {'X_train': np.array([1, 2, 3])}
    with pytest.raises(TypeError):
        save_data_splits(bad_data, tmp_path, verbose=False)


# --- load_data_splits --------------------------------------------------------
def test_load_data_splits_returns_correct_keys_and_types(tmp_path, splits):
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


def test_load_data_splits_raises_on_missing_directory():
    with pytest.raises(ValueError):
        load_data_splits('bad_path/', verbose=False)


def test_raises_on_missing_file_data(tmp_path, splits):
    splits['X_train'].to_parquet(tmp_path / 'X_train.parquet')
    with pytest.raises(FileNotFoundError):
        load_data_splits(tmp_path, verbose=False)

