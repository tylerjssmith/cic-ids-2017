"""
Test models in machine learning training pipeline.
"""
from sklearn.metrics import precision_score, recall_score, f1_score


def test_model( 
    model,
    data: dict,
    X_key: str = 'X_test', 
    y_key: str = 'y_test',
    verbose: bool = True
) -> dict:
    """
    Evaluate model performance on test data.

    Parameters
    ----------
    model
        Model to evaluate
    data : dict
        Dictionary with test data
    X_key : str, default 'X_test'
        Dictionary key for features
    y_key : str, default 'y_test'
        Dictionary key for labels
    verbose : bool, default True
        Print information

    Returns
    -------
    dict
        Performance metrics (precision, recall, F1)
    """
    if verbose:
        print('='*70)
        print('Test Model')
        print('-'*70)

    X_test = data[X_key]
    y_test = data[y_key]
    
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    if verbose:
        print(f'{"Precision:":<10} {precision:.4f}')
        print(f'{"Recall:":<10} {recall:.4f}')
        print(f'{"F1:":<10} {f1:.4f}')
        print()
    
    return {
        precision,
        recall,
        f1
    }