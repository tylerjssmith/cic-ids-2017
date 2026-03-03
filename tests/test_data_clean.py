"""Tests for data_cleaning.py"""
import pytest
import numpy as np
import pandas as pd

from data.clean import (
    drop_zero_variance, 
    remove_missing, 
    remove_infinite, 
    remove_negative
)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'col_a': [1, 1, 1],
        'col_b': [np.nan, 2, 3],
        'col_c': [1, np.inf, 3],
        'col_d': [1, 2, -3],
        'col_e': ['a', 'b', 'c']
    })


# --- drop_zero_variance ------------------------------------------------------
def test_drop_zero_variance(sample_df):
    result = drop_zero_variance(sample_df, verbose=False)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    variance = result[numeric_cols].replace([np.inf, -np.inf], np.nan).var()
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == list(range(len(result)))
    assert 'col_a' not in result.columns
    assert (variance > 0).all()


# --- remove_missing ----------------------------------------------------------
def test_remove_missing(sample_df):
    result = remove_missing(sample_df, verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == list(range(len(result)))
    assert not result.isna().any().any()


# --- remove_infinite ---------------------------------------------------------
def test_remove_infinite(sample_df):
    result = remove_infinite(sample_df, verbose=False)
    numeric_columns = result.select_dtypes(np.number).columns
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == list(range(len(result)))
    assert not np.isinf(result[numeric_columns]).any().any()


# --- remove_negative ---------------------------------------------------------
def test_remove_negative(sample_df):
    result = remove_negative(sample_df, verbose=False)
    numeric_columns = result.select_dtypes(np.number).columns
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == list(range(len(result)))
    assert not (result[numeric_columns] < 0).any().any()