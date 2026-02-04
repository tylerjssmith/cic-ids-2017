import pytest
import numpy as np
import pandas as pd
import joblib
import skops.io as sio
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from testing import evaluate_models


@pytest.fixture
def sample_data(tmp_path):
    """Create sample train/test data splits."""
    np.random.seed(42)
    
    # Create synthetic data
    X_train = pd.DataFrame(
        np.random.rand(100, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    X_test = pd.DataFrame(
        np.random.rand(30, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    # Binary labels as strings
    y_train = pd.Series(['benign'] * 50 + ['attack'] * 50)
    y_test = pd.Series(['benign'] * 15 + ['attack'] * 15)
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Save to file
    data_file = tmp_path / "data_splits.pkl"
    joblib.dump(data, data_file)
    
    return data_file, data


@pytest.fixture
def sample_models_binary(tmp_path, sample_data):
    """Create sample trained models with binary classification."""
    data_file, data = sample_data
    
    # Load data
    data_dict = joblib.load(data_file)
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    
    # Create label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Train two simple models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_pipeline = Pipeline([('classifier', lr_model)])
    lr_pipeline.fit(X_train, y_train_encoded)
    
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_pipeline = Pipeline([('classifier', rf_model)])
    rf_pipeline.fit(X_train, y_train_encoded)
    
    # Create models dictionary
    trained_models = {
        'logistic_regression': {
            'pipeline': lr_pipeline,
            'model': lr_model,
            'label_encoder': label_encoder,
            'feature_names': X_train.columns.tolist()
        },
        'random_forest': {
            'pipeline': rf_pipeline,
            'model': rf_model,
            'label_encoder': label_encoder,
            'feature_names': X_train.columns.tolist()
        }
    }
    
    # Save to skops file
    models_file = tmp_path / "models.skops"
    sio.dump(trained_models, models_file)
    
    return models_file


@pytest.fixture
def sample_models_multiclass(tmp_path):
    """Create sample trained models with multi-class classification."""
    np.random.seed(42)
    
    # Create multi-class data
    X_train = pd.DataFrame(
        np.random.rand(150, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    X_test = pd.DataFrame(
        np.random.rand(45, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    # Multi-class labels
    y_train = pd.Series(['benign'] * 50 + ['dos'] * 50 + ['portscan'] * 50)
    y_test = pd.Series(['benign'] * 15 + ['dos'] * 15 + ['portscan'] * 15)
    
    # Save data
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    data_file = tmp_path / "data_multiclass.pkl"
    joblib.dump(data, data_file)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = Pipeline([('classifier', model)])
    pipeline.fit(X_train, y_train_encoded)
    
    # Create models dictionary
    trained_models = {
        'logistic_regression': {
            'pipeline': pipeline,
            'model': model,
            'label_encoder': label_encoder,
            'feature_names': X_train.columns.tolist()
        }
    }
    
    # Save models
    models_file = tmp_path / "models_multiclass.skops"
    sio.dump(trained_models, models_file)
    
    return models_file, data_file


def test_evaluate_models_basic_functionality(sample_models_binary, sample_data):
    """Test that evaluate_models runs and returns correct structure."""
    models_file = sample_models_binary
    data_file, _ = sample_data
    
    results = evaluate_models(
        models_file=models_file,
        data_file=data_file,
        verbose=False
    )
    
    # Check structure
    assert isinstance(results, dict)
    assert 'logistic_regression' in results
    assert 'random_forest' in results
    
    # Check each model has required metrics
    for model_name, scores in results.items():
        assert 'precision' in scores
        assert 'recall' in scores
        assert 'f1' in scores
        
        # Check metrics are floats in valid range
        assert isinstance(scores['precision'], (float, np.floating))
        assert isinstance(scores['recall'], (float, np.floating))
        assert isinstance(scores['f1'], (float, np.floating))
        assert 0 <= scores['precision'] <= 1
        assert 0 <= scores['recall'] <= 1
        assert 0 <= scores['f1'] <= 1


def test_evaluate_models_subset_by_name(sample_models_binary, sample_data):
    """Test that model_names parameter correctly subsets models."""
    models_file = sample_models_binary
    data_file, _ = sample_data
    
    # Evaluate only logistic regression
    results = evaluate_models(
        models_file=models_file,
        data_file=data_file,
        model_names=['logistic_regression'],
        verbose=False
    )
    
    # Check only specified model is evaluated
    assert len(results) == 1
    assert 'logistic_regression' in results
    assert 'random_forest' not in results


def test_evaluate_models_missing_model_name_raises_error(sample_models_binary, sample_data):
    """Test that requesting non-existent model raises ValueError."""
    models_file = sample_models_binary
    data_file, _ = sample_data
    
    with pytest.raises(ValueError, match="Models not found in file"):
        evaluate_models(
            models_file=models_file,
            data_file=data_file,
            model_names=['nonexistent_model'],
            verbose=False
        )


def test_evaluate_models_multiclass(sample_models_multiclass):
    """Test that multiclass classification works correctly."""
    models_file, data_file = sample_models_multiclass
    
    results = evaluate_models(
        models_file=models_file,
        data_file=data_file,
        verbose=False
    )
    
    # Check results exist
    assert 'logistic_regression' in results
    
    # Check metrics are valid
    scores = results['logistic_regression']
    assert 0 <= scores['precision'] <= 1
    assert 0 <= scores['recall'] <= 1
    assert 0 <= scores['f1'] <= 1


def test_evaluate_models_file_not_found_errors(tmp_path):
    """Test that missing files raise appropriate errors."""
    # Test missing models file
    with pytest.raises(FileNotFoundError, match="Models file not found"):
        evaluate_models(
            models_file=tmp_path / "nonexistent_models.skops",
            data_file=tmp_path / "data.pkl",
            verbose=False
        )
    
    # Create dummy models file to test missing data file
    dummy_models = tmp_path / "dummy.skops"
    sio.dump({'dummy': {}}, dummy_models)
    
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        evaluate_models(
            models_file=dummy_models,
            data_file=tmp_path / "nonexistent_data.pkl",
            verbose=False
        )