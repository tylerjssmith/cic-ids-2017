import pytest
import numpy as np
import pandas as pd
from processing import clean_data, prepare_labels_binary, split_data


# Tests for clean_data
def test_clean_data_removes_nan():
    """Test that NaN values are removed when rm_nan=True."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, 6, 7, 8]
    })
    
    result = clean_data(
        df, 
        rm_nan=True, 
        rm_inf=False, 
        mk_flt=False, 
        verbose=False
    )
    
    assert len(result) == 3
    assert result['a'].isna().sum() == 0


def test_clean_data_removes_inf():
    """Test that infinity values are removed when rm_inf=True."""
    df = pd.DataFrame({
        'a': [1, 2, np.inf, 4],
        'b': [5, -np.inf, 7, 8]
    })
    
    result = clean_data(
        df, 
        rm_nan=False, 
        rm_inf=True, 
        mk_flt=False, 
        verbose=False
    )
    
    assert len(result) == 2
    assert not np.isinf(result['a']).any()
    assert not np.isinf(result['b']).any()


def test_clean_data_converts_int_to_float():
    """Test that integer columns are converted to float when mk_flt=True."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.0, 2.0, 3.0]
    })
    
    result = clean_data(
        df, 
        rm_nan=False, 
        rm_inf=False, 
        mk_flt=True, 
        verbose=False
    )
    
    assert result['int_col'].dtype == 'float64'
    assert result['float_col'].dtype == 'float64'


def test_clean_data_preserves_original():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, 6, 7, 8]
    })
    original_len = len(df)
    
    clean_data(df, verbose=False)
    
    assert len(df) == original_len
    assert df['a'].isna().sum() == 1


def test_clean_data_with_all_flags_disabled():
    """Test that data is unchanged when all cleaning flags are False."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, np.inf, 7, 8]
    })
    
    result = clean_data(
        df, 
        rm_nan=False, 
        rm_inf=False, 
        mk_flt=False, 
        verbose=False
    )
    
    assert len(result) == 4
    assert result['a'].dtype == df['a'].dtype


# Tests for prepare_labels_binary
def test_prepare_labels_binary_creates_binary_labels():
    """Test that binary labels are created correctly."""
    df = pd.DataFrame({
        'label': ['BENIGN', 'Attack1', 'BENIGN', 'Attack2'],
        'feature': [1, 2, 3, 4]
    })
    
    result = prepare_labels_binary(
        df, 
        label_col='label',
        benign_value='BENIGN',
        drop_original=False,
        verbose=False
    )
    
    assert 'is_attack' in result.columns
    assert list(result['is_attack']) == [0, 1, 0, 1]
    assert result['is_attack'].dtype == int


def test_prepare_labels_binary_excludes_values():
    """Test that specified values are excluded from dataset."""
    df = pd.DataFrame({
        'label': ['BENIGN', 'Attack1', 'Attack2', 'BENIGN'],
        'feature': [1, 2, 3, 4]
    })
    
    result = prepare_labels_binary(
        df,
        label_col='label',
        exclude_values=['Attack2'],
        verbose=False
    )
    
    assert len(result) == 3
    if 'label' in result.columns:
        assert 'Attack2' not in result['label'].values

def test_prepare_labels_binary_drops_original():
    """Test that original label column is dropped when drop_original=True."""
    df = pd.DataFrame({
        'label': ['BENIGN', 'Attack1'],
        'feature': [1, 2]
    })
    
    result = prepare_labels_binary(
        df,
        label_col='label',
        drop_original=True,
        verbose=False
    )
    
    assert 'label' not in result.columns
    assert 'is_attack' in result.columns


def test_prepare_labels_binary_keeps_original():
    """Test that original label column is kept when drop_original=False."""
    df = pd.DataFrame({
        'label': ['BENIGN', 'Attack1'],
        'feature': [1, 2]
    })
    
    result = prepare_labels_binary(
        df,
        label_col='label',
        drop_original=False,
        verbose=False
    )
    
    assert 'label' in result.columns
    assert 'is_attack' in result.columns


def test_prepare_labels_binary_missing_column_error():
    """Test that ValueError is raised when label column does not exist."""
    df = pd.DataFrame({
        'feature': [1, 2, 3]
    })
    
    with pytest.raises(ValueError, match="Column 'label' not found"):
        prepare_labels_binary(df, label_col='label', verbose=False)


def test_prepare_labels_binary_preserves_original():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({
        'label': ['BENIGN', 'Attack1'],
        'feature': [1, 2]
    })
    original_cols = df.columns.tolist()
    
    prepare_labels_binary(df, verbose=False)
    
    assert df.columns.tolist() == original_cols
    assert 'is_attack' not in df.columns


# Tests for split_data
def test_split_data_returns_correct_keys():
    """Test that split_data returns dictionary with correct keys."""
    df = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'is_attack': [0] * 50 + [1] * 50
    })
    
    result = split_data(df, target_col='is_attack', verbose=False)
    
    assert set(result.keys()) == {'X_train', 'X_test', 'y_train', 'y_test'}


def test_split_data_correct_split_proportion():
    """Test that data is split according to test_size parameter."""
    df = pd.DataFrame({
        'feature': range(100),
        'is_attack': [0] * 50 + [1] * 50
    })
    
    result = split_data(
        df, 
        target_col='is_attack', 
        test_size=0.2, 
        verbose=False
    )
    
    assert len(result['X_train']) == 80
    assert len(result['X_test']) == 20
    assert len(result['y_train']) == 80
    assert len(result['y_test']) == 20


def test_split_data_stratification():
    """Test that stratification maintains class balance."""
    df = pd.DataFrame({
        'feature': range(100),
        'is_attack': [0] * 80 + [1] * 20  # 80/20 imbalance
    })
    
    result = split_data(
        df, 
        target_col='is_attack',
        test_size=0.2,
        stratify_y=True,
        verbose=False
    )
    
    # Check class proportions are similar in train and test
    train_ratio = result['y_train'].sum() / len(result['y_train'])
    test_ratio = result['y_test'].sum() / len(result['y_test'])
    overall_ratio = df['is_attack'].sum() / len(df)
    
    assert abs(train_ratio - overall_ratio) < 0.05
    assert abs(test_ratio - overall_ratio) < 0.05


def test_split_data_no_target_column():
    """Test that features don't contain target column."""
    df = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'is_attack': [0] * 50 + [1] * 50
    })
    
    result = split_data(df, target_col='is_attack', verbose=False)
    
    assert 'is_attack' not in result['X_train'].columns
    assert 'is_attack' not in result['X_test'].columns


def test_split_data_missing_target_error():
    """Test that ValueError is raised when target column doesn't exist."""
    df = pd.DataFrame({
        'feature': range(100)
    })
    
    with pytest.raises(ValueError, match="Column 'is_attack' not found"):
        split_data(df, target_col='is_attack', verbose=False)


def test_split_data_reproducibility():
    """Test that same random_state produces same split."""
    df = pd.DataFrame({
        'feature': range(100),
        'is_attack': [0] * 50 + [1] * 50
    })
    
    result1 = split_data(df, random_state=42, verbose=False)
    result2 = split_data(df, random_state=42, verbose=False)
    
    pd.testing.assert_frame_equal(result1['X_train'], result2['X_train'])
    pd.testing.assert_series_equal(result1['y_train'], result2['y_train'])