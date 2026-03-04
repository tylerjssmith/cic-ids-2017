"""Tests for models/evaluate.py"""
import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from models.evaluate import evaluate_models


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def label_encoder():
    le = LabelEncoder()
    le.fit(['attack', 'benign'])
    return le


@pytest.fixture
def trained_model(label_encoder):
    X = pd.DataFrame({'f1': range(100), 'f2': range(100)})
    y = pd.Series(['attack', 'benign'] * 50)
    y_encoded = label_encoder.transform(y)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y_encoded)
    return model


@pytest.fixture
def sample_data():
    X_test = pd.DataFrame({'f1': range(20), 'f2': range(20)})
    y_test = pd.Series(['attack', 'benign'] * 10)
    return {'X_test': X_test, 'y_test': y_test}


@pytest.fixture
def sample_results(trained_model, label_encoder):
    return {
        'decision_tree': {
            'model': trained_model,
            'label_encoder': label_encoder,
            'scaler': None,
        }
    }


# --- evaluate_models ---------------------------------------------------------
def test_evaluate_models_output_structure(sample_data, sample_results):
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=sample_results),
    ):
        result = evaluate_models('results.pkl', 'data/', verbose=False)

    assert isinstance(result, dict)
    for model_name, df in result.items():
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {'class', 'precision', 'recall', 'f1_score', 'support'}


def test_evaluate_models_overall_weighted_row_present(sample_data, sample_results):
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=sample_results),
    ):
        result = evaluate_models('results.pkl', 'data/', verbose=False)

    assert 'overall_weighted' in result['decision_tree']['class'].values


def test_evaluate_models_correct_number_of_rows(sample_data, sample_results):
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=sample_results),
    ):
        result = evaluate_models('results.pkl', 'data/', verbose=False)

    n_classes = sample_data['y_test'].nunique()
    assert len(result['decision_tree']) == n_classes + 1  # +1 for overall_weighted


def test_evaluate_models_unseen_class_warning_printed(sample_data, sample_results, capsys):
    sample_data['y_test'] = pd.Series(['attack', 'benign', 'unknown'] * 6 + ['attack', 'benign'])
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=sample_results),
    ):
        evaluate_models('results.pkl', 'data/', verbose=False)

    captured = capsys.readouterr()
    assert 'unseen' in captured.out.lower()


def test_evaluate_models_unseen_class_excluded_from_results(sample_data, sample_results):
    sample_data['y_test'] = pd.Series(['attack', 'benign', 'unknown'] * 6 + ['attack', 'benign'])
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=sample_results),
    ):
        result = evaluate_models('results.pkl', 'data/', verbose=False)

    classes = result['decision_tree']['class'].values
    assert 'unknown' not in classes


def test_evaluate_models_multiple_models(sample_data, trained_model, label_encoder):
    two_models = {
        'decision_tree': {
            'model': trained_model,
            'label_encoder': label_encoder,
            'scaler': None,
        },
        'decision_tree_2': {
            'model': trained_model,
            'label_encoder': label_encoder,
            'scaler': None,
        },
    }
    with (
        patch('models.evaluate.load_data_splits', return_value=sample_data),
        patch('models.evaluate.load_results', return_value=two_models),
    ):
        result = evaluate_models('results.pkl', 'data/', verbose=False)

    assert set(result.keys()) == {'decision_tree', 'decision_tree_2'}
