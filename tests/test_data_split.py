"""Tests for data_splitting.py"""
import pytest
import pandas as pd

from data.split import split_data

# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': range(20),
        'feature2': range(20),
        'label': ['attack','benign'] * 10
    })


# --- split_data --------------------------------------------------------------
def test_split_data(sample_df):
    result = split_data(sample_df, test_size=0.2, verbose=False)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'X_train','X_test','y_train','y_test'}
    assert len(result['X_train']) == len(result['y_train']) == 16
    assert len(result['X_test']) == len(result['y_test']) == 4
    assert 'label' not in result['X_train'].columns
    assert 'label' not in result['X_test'].columns


def test_split_data_raises_on_label_not_found(sample_df):
    df = sample_df.drop(columns='label')
    with pytest.raises(ValueError):
        split_data(df, label_col='label', verbose=False)


def test_split_data_stratification(sample_df):
    result = split_data(sample_df, test_size=0.2, stratify=True, verbose=False)
    assert result['y_train'].value_counts()['attack'] == 8
    assert result['y_test'].value_counts()['attack'] == 2


def test_split_data_reproducible(sample_df):
    result1 = split_data(sample_df, random_state=42, verbose=False)
    result2 = split_data(sample_df, random_state=42, verbose=False)
    pd.testing.assert_frame_equal(result1['X_train'], result2['X_train'])
    