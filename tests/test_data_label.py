"""Tests for data_labeling.py"""
import pytest
import pandas as pd

from data.label import prepare_labels


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'label': ['BENIGN', 'BENIGN', 'DoS Hulk', 'Heartbleed'],
    })


# --- prepare_labels ----------------------------------------------------------
def test_prepare_labels_drop(sample_df):
    result = prepare_labels(sample_df, 
                            drop_labels=['Heartbleed'],
                            replace_labels=None, 
                            clean_labels=False, 
                            verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'Heartbleed' not in result['label'].str.lower().values


def test_prepare_labels_replace(sample_df):
    result = prepare_labels(sample_df,
                            drop_labels=None,
                            replace_labels={'DoS Hulk': 'denial_of_service'},
                            clean_labels=False,
                            verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert 'denial_of_service' in result['label'].str.lower().values


def test_prepare_labels_clean(sample_df):
    result = prepare_labels(sample_df, 
                            drop_labels=None,
                            replace_labels=None,
                            clean_labels=True, 
                            verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result['label']) == [
        'benign', 
        'benign', 
        'dos_hulk', 
        'heartbleed'
    ]


def test_prepare_labels_raises_error_on_raw_label_column(sample_df):
    sample_df['_raw_label'] = 1
    with pytest.raises(ValueError):
        prepare_labels(sample_df, 
                       drop_labels=None,
                       replace_labels=None,
                       clean_labels=True, 
                       verbose=False)
