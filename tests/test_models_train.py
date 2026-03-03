import pytest
import pandas as pd

from models.train import train_models

# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_data():
    X = pd.DataFrame({'f1': range(100), 'f2': range(100)})
    y = pd.Series(['attack', 'benign'] * 50)
    return {'X_train': X, 'y_train': y}

@pytest.fixture
def sample_models():
    return {
        'decision_tree': {
            'module': 'sklearn.tree',
            'class': 'DecisionTreeClassifier',
            'hyperparameters': {'max_depth': 3}
        }
    }


# --- train_models ------------------------------------------------------------
def test_train_models_result_structure(sample_data, sample_models):
    result = train_models(
        sample_data, sample_models, 
        cv_k=2, 
        verbose=False
    )
    assert isinstance(result, dict)
    for model_result in result.values():
        assert set(model_result.keys()) == {
            'model', 'scaler', 'label_encoder', 'metrics',
            'precision_mean', 'precision_std', 'recall_mean',
            'recall_std', 'f1_mean', 'f1_std', 'y_pred_proba', 'y_encoded'
        }
        assert isinstance(model_result['metrics'], pd.DataFrame)


def test_train_models_raises_error_smote_weights(sample_data, sample_models):
    with pytest.raises(ValueError):
        train_models(
            data=sample_data, 
            models=sample_models, 
            use_smote=True, 
            use_class_weights=True
        )


def test_train_models_cv_kwargs_ignores_cv(sample_data, sample_models):
    result = train_models(
        sample_data, sample_models,
        cv_k=2,
        cv_kwargs={'cv': 99},
        verbose=False
    )
    assert isinstance(result, dict)


def test_train_models_bad_module_prints_error(sample_data, sample_models, capsys):
    sample_models['decision_tree']['module'] = 'bad_module'
    train_models(sample_data, sample_models, cv_k=2, verbose=False)
    captured = capsys.readouterr()
    assert 'Error loading decision_tree' in captured.out


def test_train_models_bad_class_prints_error(sample_data, sample_models, capsys):
    sample_models['decision_tree']['class'] = 'BadClass'
    train_models(sample_data, sample_models, cv_k=2, verbose=False)
    captured = capsys.readouterr()
    assert 'Error loading decision_tree' in captured.out


def test_train_models_bad_hyperparameters_prints_error(sample_data, sample_models, capsys):
    sample_models['decision_tree']['hyperparameters'] = {'bad_param': 99}
    train_models(sample_data, sample_models, cv_k=2, verbose=False)
    captured = capsys.readouterr()
    assert 'Error initializing decision_tree' in captured.out


def test_train_models_scaler_none_by_default(sample_data, sample_models):
    result = train_models(sample_data, sample_models, cv_k=2, verbose=False)
    assert result['decision_tree']['scaler'] is None


def test_train_models_y_pred_proba_shape(sample_data, sample_models):
    result = train_models(sample_data, sample_models, cv_k=2, verbose=False)
    n_samples = len(sample_data['X_train'])
    n_classes = sample_data['y_train'].nunique()
    assert result['decision_tree']['y_pred_proba'].shape == (n_samples, n_classes)