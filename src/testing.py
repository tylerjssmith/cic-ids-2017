"""
Test models in machine learning training pipeline.
"""
import skops.io as sio
import joblib
from typing import Union
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd


def evaluate_models(
    models_file: Union[str, Path],
    data_file: Union[str, Path],
    model_names: list = None,
    skops_trusted: list = ['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'],
    verbose: bool = True
) -> dict:
    """
    Evaluate trained models on test data.
    
    Parameters
    ----------
    models_file : str
        Path to skops file containing trained models
    data_file : str
        Path to pickle file containing data splits
    model_names : list
        Models in models_file to include
    skops_trusted : list
        Trusted skops objects
    verbose : bool, default True
        Print evaluation metrics
        
    Returns
    -------
    dict
        Dictionary with evaluation scores for each model
        Structure: {model_name: {'precision': float, 'recall': float, 'f1': float}}
    """
    if verbose:
        print("="*70)
        print("Evaluate Models")
        print("-"*70)

    # Load models
    models_file = Path(models_file)
    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    trained_models = sio.load(models_file, trusted=skops_trusted)

    # Subset models (optional)
    if model_names is not None:
        missing = [name for name in model_names if name not in trained_models]
        if missing:
            raise ValueError(f"Models not found in file: {missing}")
        
        trained_models = {key: trained_models[key] for key in model_names}
    
    # Load data
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = joblib.load(data_file)

    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    if verbose:
        print(f"Loaded models from: {models_file}")
        print(f"Loaded data from: {data_file}")
        print()
    
    # Evaluate each model
    results = {}
    
    for model_name, model_info in trained_models.items():
        # Encode test labels
        label_encoder = model_info.get('label_encoder')
        if label_encoder is None:
            raise ValueError(f"Label encoder not found with {model_name}")
            
        y_test_encoded = label_encoder.transform(y_test)
        
        # Determine if binary or multi-class
        n_classes = len(label_encoder.classes_)
        is_binary = n_classes == 2
        
        if verbose:       
            print("-"*70)
            print(f"Model: {model_name}")
            print("-"*70)
        
        # Predict y
        pipeline = model_info['pipeline']        
        y_pred_encoded = pipeline.predict(X_test)
        
        # Calculate Scores
        average = None if is_binary else 'weighted'
        
        precision_raw = precision_score(
            y_test_encoded, 
            y_pred_encoded, 
            average=average, 
            zero_division=0
        )
        recall_raw = recall_score(
            y_test_encoded, 
            y_pred_encoded, 
            average=average, 
            zero_division=0
        )
        f1_raw = f1_score(
            y_test_encoded, 
            y_pred_encoded, 
            average=average, 
            zero_division=0
        )
                
        # Store results
        if is_binary:
            precision = precision_raw[1]  # Positive class score
            recall = recall_raw[1]
            f1 = f1_raw[1]
        else:
            precision = precision_raw
            recall = recall_raw
            f1 = f1_raw
    
        results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if verbose:
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print()
            
            # Decode predictions for classification report
            y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
            y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)
            
            print("Classification Report:")
            print(classification_report(y_test_decoded, y_pred_decoded, 
                                       target_names=label_encoder.classes_, 
                                       zero_division=0))
            print()
    
    if verbose:
        print("-"*70)
        print("Summary of Results")
        print("-"*70)
        
        summary_data = []
        for model_name, scores in results.items():
            summary_data.append({
                'Model': model_name,
                'Precision': f"{scores['precision']:.4f}",
                'Recall': f"{scores['recall']:.4f}",
                'F1-Score': f"{scores['f1']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print()
    
    return results