import pytest
from main import load_function


def test_load_function():
    """Test that load_function imports and returns a function correctly."""
    func = load_function('utilities', 'load_data')
    
    assert callable(func)
    assert func.__name__ == 'load_data'


def test_load_function_invalid_module():
    """Test that load_function raises error for non-existent module."""
    with pytest.raises(ModuleNotFoundError):
        load_function('nonexistent_module', 'load_data')


def test_load_function_invalid_function():
    """Test that load_function raises error for non-existent function."""
    with pytest.raises(AttributeError):
        load_function('utilities', 'nonexistent_function')