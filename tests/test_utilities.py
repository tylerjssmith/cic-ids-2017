import pytest
import pandas as pd
import joblib
from pathlib import Path
from utilities import load_data, save_data_splits


def test_load_csv(tmp_path):
    """Test that load_data successfully loads CSV files."""
    df_original = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['Seattle', 'Baltimore', 'New York']
    })
    
    csv_file = tmp_path / "test_data.csv"
    df_original.to_csv(csv_file, index=False)
    df_loaded = load_data(
        csv_file, 
        data_type='csv', 
        clean_columns=False, 
        verbose=False
    )
    
    assert df_loaded.shape == (3, 3)
    pd.testing.assert_frame_equal(df_loaded, df_original)


def test_load_parquet(tmp_path):
    """Test that load_data successfully loads Parquet files."""
    df_original = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['Seattle', 'Baltimore', 'New York']
    })
    
    parquet_file = tmp_path / "test_data.parquet"
    df_original.to_parquet(parquet_file, index=False)
    df_loaded = load_data(
        parquet_file, 
        data_type='parquet', 
        clean_columns=False, 
        verbose=False
    )
    
    assert df_loaded.shape == (3, 3)
    pd.testing.assert_frame_equal(df_loaded, df_original)


def test_load_data_column_cleaning(tmp_path):
    """Test that column names are cleaned properly."""
    df_original = pd.DataFrame({
        ' First Name ': ['Alice', 'Bob'],
        'Last Name': ['Smith', 'Jones'],
        'AGE': [25, 30],
        'bytes/s': [1,2]
    })
    
    csv_file = tmp_path / "test_data.csv"
    df_original.to_csv(csv_file, index=False)
    df_loaded = load_data(
        csv_file, 
        data_type='csv', 
        clean_columns=True, 
        verbose=False
    )
    
    expected_columns = ['first_name', 'last_name', 'age', 'bytes_s']
    assert list(df_loaded.columns) == expected_columns


def test_load_data_invalid_type(tmp_path):
    """Test that invalid data_type raises ValueError."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({'col': [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    with pytest.raises(ValueError, match="data_type must be 'csv' or 'parquet'"):
        load_data(csv_file, data_type='excel', verbose=False)


def test_save_data_splits_roundtrip(tmp_path):
    """Test that saved data can be loaded back correctly."""
    original_data = {
        'train': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
        'test': pd.DataFrame({'x': [7, 8], 'y': [9, 10]})
    }
    
    filepath = tmp_path / "data_splits.joblib"
    save_data_splits(original_data, filepath, verbose=False)
    
    loaded_data = joblib.load(filepath)
    
    assert 'train' in loaded_data
    assert 'test' in loaded_data
    pd.testing.assert_frame_equal(loaded_data['train'], original_data['train'])
    pd.testing.assert_frame_equal(loaded_data['test'], original_data['test'])


def test_save_data_splits_creates_directories(tmp_path):
    """Test that parent directories are created if they do not exist."""
    data = {'train': pd.DataFrame({'x': [1, 2, 3]})}
    filepath = tmp_path / "nested" / "subdirectory" / "data.joblib"
    
    save_data_splits(data, filepath, verbose=False)
    
    assert filepath.exists()
    assert filepath.parent.exists()