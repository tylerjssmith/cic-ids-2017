"""Tests for models/finalize.py"""
import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from models.finalize import finalize_model


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_data():
    X = pd.DataFrame({'f1': range(10), 'f2': range(10)})
    y = pd.Series(['attack', 'benign'] * 5)
    return {
        'X_train': X.iloc[:8].reset_index(drop=True),
        'X_test':  X.iloc[8:].reset_index(drop=True),
        'y_train': y.iloc[:8].reset_index(drop=True),
        'y_test':  y.iloc[8:].reset_index(drop=True),
    }


@pytest.fixture
def sample_results(sample_data):
    le = LabelEncoder()
    y = pd.concat([sample_data['y_train'], sample_data['y_test']])
    le.fit(y)
    y_encoded = le.transform(sample_data['y_train'])

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(sample_data['X_train'], y_encoded)

    return {
        'decision_tree': {
            'model': model,
            'label_encoder': le,
            'scaler': None,
        }
    }


# --- finalize_model ----------------------------------------------------------
def test_finalize_model_happy_path(sample_data, sample_results):
    with (
        patch('models.finalize.load_data_splits', return_value=sample_data),
        patch('models.finalize.load_results', return_value=sample_results),
    ):
        result = finalize_model(
            results='results.pkl',
            data='data/',
            model_name='decision_tree',
            verbose=False,
        )

    assert set(result.keys()) == {'model', 'label_encoder', 'feature_names', 'scaler', 'metadata'}
    assert result['feature_names'] == ['f1', 'f2']
    assert result['scaler'] is None

    metadata = result['metadata']
    assert metadata['model_name'] == 'decision_tree'
    assert metadata['model_class'] == 'DecisionTreeClassifier'
    assert metadata['n_samples'] == 10
    assert metadata['n_features'] == 2
    assert metadata['n_classes'] == 2
    assert metadata['scaling_used'] is False
    assert hasattr(result['model'], 'predict')


def test_finalize_model_raises_on_missing_model_name(sample_data, sample_results):
    with (
        patch('models.finalize.load_data_splits', return_value=sample_data),
        patch('models.finalize.load_results', return_value=sample_results),
    ):
        with pytest.raises(KeyError, match='bad_model'):
            finalize_model(
                results='results.pkl',
                data='data/',
                model_name='bad_model',
                verbose=False,
            )
