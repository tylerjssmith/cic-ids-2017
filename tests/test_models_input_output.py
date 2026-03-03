"""Tests for model_io.py"""
import pytest
import joblib
import pandas as pd
import skops.io as sio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models.input_output import (
    load_results, 
    save_results, 
    save_model, 
    _hash_file
)

# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def sample_result():
    return {
        'accuracy': 0.99, 
        'model': 'RandomForest'
    }


@pytest.fixture
def sample_bundle(tmp_path):
    X = pd.DataFrame({'f1': range(100), 'f2': range(100)})
    y = pd.Series(['attack', 'benign'] * 50)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(pd.DataFrame(X_scaled, columns=X.columns), y_encoded)

    return {
        'model': model,
        'label_encoder': le,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'metadata': {'description': 'test bundle'}
    }


# --- save_results ------------------------------------------------------------
def test_save_results(tmp_path, sample_result):
    save_results(sample_result, 'results.pkl', tmp_path, verbose=False)
    loaded = joblib.load(tmp_path / 'results.pkl')
    assert loaded == sample_result


# --- load_results ------------------------------------------------------------
def test_load_results(tmp_path, sample_result):
    joblib.dump(sample_result, tmp_path / 'results.pkl')
    loaded = load_results(tmp_path / 'results.pkl', verbose=False)
    assert loaded == sample_result


def test_load_results_raises_on_missing_file_results():
    with pytest.raises(FileNotFoundError):
        load_results('nonexistent.pkl', verbose=False)


# --- save_models -------------------------------------------------------------
def test_save_model(sample_bundle, tmp_path):
    save_model(sample_bundle, tmp_path / 'bundle.skops', verbose=False)
    loaded = sio.load(
        tmp_path / 'bundle.skops',
        trusted=sio.get_untrusted_types(file=tmp_path / 'bundle.skops')
    )
    assert list(loaded['label_encoder'].classes_) == list(sample_bundle['label_encoder'].classes_)
    assert loaded['feature_names'] == sample_bundle['feature_names']
    assert loaded['metadata'] == sample_bundle['metadata']


# Argument Checks
def test_save_model_raises_on_missing_keys(sample_bundle, tmp_path):
    del sample_bundle['metadata']
    with pytest.raises(ValueError):
        save_model(sample_bundle, 
                   tmp_path / 'bundle.skops', 
                   verbose=False)


def test_save_model_raises_on_wrong_extension(sample_bundle, tmp_path):
    with pytest.raises(ValueError):
        save_model(sample_bundle, 
                   tmp_path / 'bundle.wrong', 
                   verbose=False)


def test_save_model_raises_on_invalid_compresslevel(sample_bundle, tmp_path):
    with pytest.raises(ValueError):
        save_model(sample_bundle, 
                   tmp_path / 'bundle.skops', 
                   compresslevel=10, 
                   verbose=False)


# File Hash
def test_save_model_creates_hash_file(sample_bundle, tmp_path):
    save_model(sample_bundle, tmp_path / 'model.skops', verbose=False)
    assert (tmp_path / 'model.sha256').exists()


def test_save_model_hash_is_valid_sha256(sample_bundle, tmp_path):
    save_model(sample_bundle, tmp_path / 'model.skops', verbose=False)
    file_hash = (tmp_path / 'model.sha256').read_text()
    assert len(file_hash) == 64
    assert all(c in '0123456789abcdef' for c in file_hash)


def test_save_model_hash_matches_file(sample_bundle, tmp_path):
    save_model(sample_bundle, tmp_path / 'model.skops', verbose=False)
    saved_hash = (tmp_path / 'model.sha256').read_text()
    recomputed_hash = _hash_file(tmp_path / 'model.skops')
    assert saved_hash == recomputed_hash


def test_save_model_returns_hash(sample_bundle, tmp_path):
    file_hash = save_model(sample_bundle, tmp_path / 'model.skops', verbose=False)
    saved_hash = (tmp_path / 'model.sha256').read_text()
    assert file_hash == saved_hash


# Smoke Test
def test_save_model_smoke_test_raises_on_bad_model(sample_bundle, tmp_path):
    sample_bundle['model'] = DecisionTreeClassifier()  # unfitted
    with pytest.raises(ValueError, match='smoke test'):
        save_model(sample_bundle, tmp_path / 'model.skops', verbose=False)