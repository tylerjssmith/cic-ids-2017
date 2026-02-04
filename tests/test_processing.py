"""
Tests for processing.py module.
"""
import pytest
import numpy as np
import pandas as pd
from processing import prepare_labels, drop_features, clean_data, split_data


@pytest.fixture
def sample_labels_df():
    """Create sample DataFrame with various labels."""
    return pd.DataFrame({
        'Label': ['BENIGN', 'DoS Hulk', 'DoS slowloris', 'DDoS', 'Bot', 
                  'BENIGN', 'DoS Hulk', 'PortScan', 'BENIGN', 'Bot'],
        'feature1': range(10),
        'feature2': range(10, 20)
    })


def test_prepare_labels_basic_functionality(sample_labels_df):
    """Test basic label preparation without any transformations."""
    result = prepare_labels(
        df=sample_labels_df,
        label_col='Label',
        exclude_values=None,
        replace_values=None,
        clean_values=False,
        verbose=False
    )
    
    assert 'label' in result.columns
    assert 'Label' not in result.columns
    assert len(result) == len(sample_labels_df)
    assert set(result['label']) == set(sample_labels_df['Label'])


def test_prepare_labels_exclude_values(sample_labels_df):
    """Test that exclude_values correctly removes rows."""
    result = prepare_labels(
        df=sample_labels_df,
        label_col='Label',
        exclude_values=['Bot', 'PortScan'],
        replace_values=None,
        clean_values=False,
        verbose=False
    )
    
    # Should remove 3 rows (2 Bot + 1 PortScan)
    assert len(result) == 7
    assert 'Bot' not in result['label'].values
    assert 'PortScan' not in result['label'].values
    assert 'BENIGN' in result['label'].values


def test_prepare_labels_replace_values_with_lists(sample_labels_df):
    """Test replace_values with dictionary mapping new -> list of old values."""
    replace_map = {
        0: ['BENIGN'],
        1: ['DoS Hulk', 'DoS slowloris', 'DDoS']
    }
    
    result = prepare_labels(
        df=sample_labels_df,
        label_col='Label',
        exclude_values=None,
        replace_values=replace_map,
        clean_values=False,
        verbose=False
    )
    
    assert 0 in result['label'].values
    assert 1 in result['label'].values
    assert 'BENIGN' not in result['label'].values
    assert 'DoS Hulk' not in result['label'].values


def test_prepare_labels_clean_values(sample_labels_df):
    """Test that clean_values correctly normalizes label names."""
    result = prepare_labels(
        df=sample_labels_df,
        label_col='Label',
        exclude_values=None,
        replace_values=None,
        clean_values=True,
        verbose=False
    )
    
    assert 'benign' in result['label'].values
    assert 'dos_hulk' in result['label'].values
    assert 'dos_slowloris' in result['label'].values
    assert 'ddos' in result['label'].values
    
    for label in result['label'].unique():
        assert label == label.lower()
        assert ' ' not in label


def test_prepare_labels_combined_operations(sample_labels_df):
    """Test combining exclude, replace, and clean operations."""
    replace_map = {
        'benign': ['BENIGN'],
        'attack': ['DoS Hulk', 'DoS slowloris', 'DDoS']
    }
    
    result = prepare_labels(
        df=sample_labels_df,
        label_col='Label',
        exclude_values=['Bot', 'PortScan'],
        replace_values=replace_map,
        clean_values=True,
        verbose=False
    )
    
    # Should have 7 rows (excluded 3)
    assert len(result) == 7
    
    # Should have only 2 unique labels
    assert set(result['label']) == {'benign', 'attack'}
    
    # Check counts
    assert (result['label'] == 'benign').sum() == 3
    assert (result['label'] == 'attack').sum() == 4


@pytest.fixture
def sample_features_df():
    """Create sample DataFrame with various feature types."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'zero_var': [5, 5, 5, 5, 5],  # Zero variance
        'feature3': [1.1, 2.2, 3.3, 4.4, 5.5],
        'label': ['a', 'b', 'a', 'b', 'a']
    })


def test_drop_features_drop_named_columns(sample_features_df):
    """Test dropping specific named columns."""
    result = drop_features(
        df=sample_features_df,
        drop=['feature1', 'feature3'],
        rm_zv=False,
        verbose=False
    )
    
    assert 'feature1' not in result.columns
    assert 'feature3' not in result.columns
    assert 'feature2' in result.columns
    assert 'zero_var' in result.columns


def test_drop_features_zero_variance(sample_features_df):
    """Test removal of zero-variance columns."""
    result = drop_features(
        df=sample_features_df,
        drop=[],
        rm_zv=True,
        verbose=False
    )
    
    assert 'zero_var' not in result.columns
    assert 'feature1' in result.columns
    assert 'feature2' in result.columns


def test_drop_features_ignores_non_numeric_for_variance(sample_features_df):
    """Test that non-numeric columns are not dropped for zero variance."""
    result = drop_features(
        df=sample_features_df,
        drop=[],
        rm_zv=True,
        verbose=False
    )
    
    # Label column (string) should remain even though we check variance
    assert 'label' in result.columns


def test_drop_features_combined_operations(sample_features_df):
    """Test combining drop and zero-variance removal."""
    result = drop_features(
        df=sample_features_df,
        drop=['feature1'],
        rm_zv=True,
        verbose=False
    )
    
    assert 'feature1' not in result.columns  # Explicitly dropped
    assert 'zero_var' not in result.columns  # Zero variance
    assert 'feature2' in result.columns
    assert 'feature3' in result.columns


def test_drop_features_empty_drop_list(sample_features_df):
    """Test that empty drop list works correctly."""
    result = drop_features(
        df=sample_features_df,
        drop=[],
        rm_zv=False,
        verbose=False
    )
    
    # All columns should remain
    assert len(result.columns) == len(sample_features_df.columns)


@pytest.fixture
def sample_dirty_df():
    """Create sample DataFrame with various data quality issues."""
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, np.inf, 30, -np.inf, 50],
        'feature3': [1.1, 2.2, -3.3, 4.4, 5.5],
        'label': ['a', 'b', 'a', 'b', 'a']
    })


def test_clean_data_remove_nan(sample_dirty_df):
    """Test removal of NaN values."""
    result = clean_data(
        df=sample_dirty_df,
        rm_nan=True,
        rm_inf=False,
        rm_neg=False,
        verbose=False
    )
    
    # Should remove 1 row with NaN
    assert len(result) == 4
    assert not result['feature1'].isna().any()


def test_clean_data_remove_inf(sample_dirty_df):
    """Test removal of infinity values."""
    result = clean_data(
        df=sample_dirty_df,
        rm_nan=False,
        rm_inf=True,
        rm_neg=False,
        verbose=False
    )
    
    # Should remove 2 rows with inf and -inf
    assert len(result) == 3
    assert not np.isinf(result['feature2']).any()


def test_clean_data_remove_negative(sample_dirty_df):
    """Test removal of negative values."""
    result = clean_data(
        df=sample_dirty_df,
        rm_nan=False,
        rm_inf=False,
        rm_neg=True,
        verbose=False
    )
    
    # Should remove row with -3.3 (and -inf if present)
    assert len(result) == 3
    assert (result['feature3'] >= 0).all()


def test_clean_data_all_cleaning_operations(sample_dirty_df):
    """Test all cleaning operations combined."""
    result = clean_data(
        df=sample_dirty_df,
        rm_nan=True,
        rm_inf=True,
        rm_neg=True,
        verbose=False
    )
    
    # Should remove rows with: NaN (1), inf (1), -inf (1), negative (1)
    # Some rows may have multiple issues
    assert len(result) <= len(sample_dirty_df)
    assert not result.isna().any().any()
    assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    
    # Check all numeric values are >= 0
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert (result[numeric_cols] >= 0).all().all()


def test_clean_data_preserves_clean_data():
    """Test that clean data remains unchanged."""
    clean_df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'label': ['a', 'b', 'a', 'b', 'a']
    })
    
    result = clean_data(
        df=clean_df,
        rm_nan=True,
        rm_inf=True,
        rm_neg=True,
        verbose=False
    )
    
    # Should be identical
    assert len(result) == len(clean_df)
    pd.testing.assert_frame_equal(result, clean_df)


@pytest.fixture
def sample_split_df():
    """Create sample DataFrame for splitting."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'label': ['benign'] * 50 + ['attack'] * 50
    })


def test_split_data_basic_functionality(sample_split_df):
    """Test basic train-test split."""
    result = split_data(
        df=sample_split_df,
        target_col='label',
        test_size=0.2,
        stratify_y=True,
        random_state=42,
        verbose=False
    )
    
    # Check structure
    assert isinstance(result, dict)
    assert 'X_train' in result
    assert 'X_test' in result
    assert 'y_train' in result
    assert 'y_test' in result
    
    # Check sizes
    assert len(result['X_train']) == 80
    assert len(result['X_test']) == 20
    assert len(result['y_train']) == 80
    assert len(result['y_test']) == 20


def test_split_data_stratification(sample_split_df):
    """Test that stratification maintains class balance."""
    result = split_data(
        df=sample_split_df,
        target_col='label',
        test_size=0.2,
        stratify_y=True,
        random_state=42,
        verbose=False
    )
    
    # Check train set balance (should be ~50/50)
    train_counts = result['y_train'].value_counts()
    assert abs(train_counts['benign'] - 40) <= 1  # Allow small variance
    assert abs(train_counts['attack'] - 40) <= 1
    
    # Check test set balance (should be ~50/50)
    test_counts = result['y_test'].value_counts()
    assert abs(test_counts['benign'] - 10) <= 1
    assert abs(test_counts['attack'] - 10) <= 1


def test_split_data_no_stratification():
    """Test splitting without stratification."""
    # Create imbalanced dataset
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'label': ['benign'] * 90 + ['attack'] * 10
    })
    
    result = split_data(
        df=df,
        target_col='label',
        test_size=0.2,
        stratify_y=False,
        random_state=42,
        verbose=False
    )
    
    # Should still create valid splits
    assert len(result['X_train']) == 80
    assert len(result['X_test']) == 20


def test_split_data_custom_test_size(sample_split_df):
    """Test custom test_size parameter."""
    result = split_data(
        df=sample_split_df,
        target_col='label',
        test_size=0.3,
        random_state=42,
        verbose=False
    )
    
    assert len(result['X_train']) == 70
    assert len(result['X_test']) == 30


def test_split_data_missing_target_column_raises_error(sample_split_df):
    """Test that missing target column raises ValueError."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        split_data(
            df=sample_split_df,
            target_col='nonexistent',
            verbose=False
        )


def test_split_data_reproducibility(sample_split_df):
    """Test that same random_state produces same split."""
    result1 = split_data(
        df=sample_split_df,
        target_col='label',
        random_state=42,
        verbose=False
    )
    
    result2 = split_data(
        df=sample_split_df,
        target_col='label',
        random_state=42,
        verbose=False
    )
    
    # Should produce identical splits
    pd.testing.assert_frame_equal(result1['X_train'], result2['X_train'])
    pd.testing.assert_frame_equal(result1['X_test'], result2['X_test'])
    pd.testing.assert_series_equal(result1['y_train'], result2['y_train'])
    pd.testing.assert_series_equal(result1['y_test'], result2['y_test'])