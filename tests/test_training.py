import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from training import train_models


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    return {
        'X_train': X_train,
        'y_train': y_train
    }


@pytest.fixture
def simple_models_config():
    """Create simple model configuration."""
    return {
        'logistic_regression': {
            'module': 'sklearn.linear_model',
            'class': 'LogisticRegression',
            'hyperparameters': {
                'max_iter': 1000,
                'random_state': 42
            },
            'scale_features': False
        }
    }


@pytest.fixture
def multiple_models_config():
    """Create configuration with multiple models."""
    return {
        'logistic_regression': {
            'module': 'sklearn.linear_model',
            'class': 'LogisticRegression',
            'hyperparameters': {
                'max_iter': 1000,
                'random_state': 42
            },
            'scale_features': True
        },
        'decision_tree': {
            'module': 'sklearn.tree',
            'class': 'DecisionTreeClassifier',
            'hyperparameters': {
                'max_depth': 5,
                'random_state': 42
            },
            'scale_features': False
        }
    }


@patch('training.mlflow')
def test_train_models_returns_correct_structure(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that train_models returns dictionary with correct structure."""
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    assert 'logistic_regression' in result
    assert 'pipeline' in result['logistic_regression']
    assert 'model' in result['logistic_regression']
    assert 'scaler' in result['logistic_regression']
    assert 'cv_mean' in result['logistic_regression']
    assert 'cv_std' in result['logistic_regression']


@patch('training.mlflow')
def test_train_models_pipeline_has_classifier(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that returned pipeline contains classifier."""
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    pipeline = result['logistic_regression']['pipeline']
    assert 'classifier' in pipeline.named_steps
    assert isinstance(
        pipeline.named_steps['classifier'],
        LogisticRegression
    )


@patch('training.mlflow')
def test_train_models_with_scaling(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that scaler is added when scale_features=True."""
    simple_models_config['logistic_regression']['scale_features'] = True
    
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    pipeline = result['logistic_regression']['pipeline']
    assert 'scaler' in pipeline.named_steps
    assert result['logistic_regression']['scaler'] is not None


@patch('training.mlflow')
def test_train_models_without_scaling(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that scaler is not added when scale_features=False."""
    simple_models_config['logistic_regression']['scale_features'] = False
    
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    pipeline = result['logistic_regression']['pipeline']
    assert 'scaler' not in pipeline.named_steps
    assert result['logistic_regression']['scaler'] is None


@patch('training.mlflow')
def test_train_models_cv_metrics_present(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that cross-validation metrics are calculated."""
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    cv_mean = result['logistic_regression']['cv_mean']
    cv_std = result['logistic_regression']['cv_std']
    
    assert 'precision' in cv_mean
    assert 'recall' in cv_mean
    assert 'f1' in cv_mean
    assert 'precision' in cv_std
    assert 'recall' in cv_std
    assert 'f1' in cv_std
    
    # Check values are in reasonable range
    assert 0 <= cv_mean['precision'] <= 1
    assert 0 <= cv_mean['recall'] <= 1
    assert 0 <= cv_mean['f1'] <= 1


@patch('training.mlflow')
def test_train_models_pipeline_is_fitted(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that returned pipeline is fitted and can make predictions."""
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    pipeline = result['logistic_regression']['pipeline']
    
    # Should be able to predict without error
    predictions = pipeline.predict(sample_data['X_train'])
    assert len(predictions) == len(sample_data['X_train'])
    assert all(pred in [0, 1] for pred in predictions)


@patch('training.mlflow')
def test_train_models_multiple_models(
    mock_mlflow,
    sample_data,
    multiple_models_config
):
    """Test training multiple models at once."""
    result = train_models(
        data=sample_data,
        models_config=multiple_models_config,
        cv=3,
        verbose=False
    )
    
    assert len(result) == 2
    assert 'logistic_regression' in result
    assert 'decision_tree' in result
    
    # Check both are fitted
    lr_pred = result['logistic_regression']['pipeline'].predict(
        sample_data['X_train']
    )
    dt_pred = result['decision_tree']['pipeline'].predict(
        sample_data['X_train']
    )
    assert len(lr_pred) == len(sample_data['X_train'])
    assert len(dt_pred) == len(sample_data['X_train'])


@patch('training.mlflow')
@patch('training.joblib.dump')
def test_train_models_saves_to_output_dir(
    mock_dump,
    mock_mlflow,
    sample_data,
    simple_models_config,
    tmp_path
):
    """Test that models are saved when output_dir is provided."""
    output_dir = tmp_path / "models"
    
    train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        output_dir=str(output_dir),
        verbose=False
    )
    
    # Check joblib.dump was called
    assert mock_dump.called
    call_args = mock_dump.call_args[0]
    saved_path = call_args[1]
    
    assert 'logistic_regression.pkl' in str(saved_path)


@patch('training.mlflow')
def test_train_models_hyperparameters_applied(
    mock_mlflow,
    sample_data,
    simple_models_config
):
    """Test that hyperparameters are applied to the model."""
    result = train_models(
        data=sample_data,
        models_config=simple_models_config,
        cv=3,
        verbose=False
    )
    
    model = result['logistic_regression']['model']
    assert model.max_iter == 1000
    assert model.random_state == 42


@patch('training.mlflow')
def test_train_models_config_without_hyperparameters(
    mock_mlflow,
    sample_data
):
    """Test that model works when hyperparameters not specified."""
    config = {
        'logistic_regression': {
            'module': 'sklearn.linear_model',
            'class': 'LogisticRegression',
            'scale_features': False
        }
    }
    
    result = train_models(
        data=sample_data,
        models_config=config,
        cv=3,
        verbose=False
    )
    
    assert 'logistic_regression' in result
    assert result['logistic_regression']['pipeline'] is not None