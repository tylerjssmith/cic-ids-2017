"""Tests for test_data_feature_engineering.py."""
import pytest
import pandas as pd

from data.build_features import (
    drop_features, 
    indicate_service
)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'destination_port': [20, 21, 22, 80, 443],
        'label': ['BENIGN', 'DoS', 'BENIGN', 'DoS', 'DoS'],
    })

@pytest.fixture
def service_port_map():
    return {
        'ftp': [20, 21],
        'ssh': [22],
        'http': [80, 443]
    }


# --- drop_features -----------------------------------------------------------
def test_drop_features_removes_named_columns(sample_df):
    result = drop_features(sample_df, drop=['label'], verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert 'label' not in result.columns
    assert len(result.columns) == len(sample_df.columns) - 1


def test_drop_features_raises_on_missing_column(sample_df):
    with pytest.raises(ValueError):
        drop_features(sample_df, drop=['missing_column'], verbose=False)


# --- indicate_service --------------------------------------------------------
def test_indicate_service(sample_df, service_port_map):
    result = indicate_service(sample_df, 
                              service_port_map=service_port_map, 
                              port_column='destination_port',
                              verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert 'destination_port' not in result.columns
    assert 'is_ftp' in result.columns
    assert 'is_ssh' in result.columns
    assert 'is_http' in result.columns
    assert len(result[result['is_ftp'] == 1]) == 2
    assert len(result[result['is_ssh'] == 1]) == 1
    assert len(result[result['is_http'] == 1]) == 2

