"""
Run machine learning training pipeline for network intrusion detection.
"""
import argparse
import importlib
import yaml


def load_function(module_path: str, function_name: str):
    """Dynamically import and return a function."""
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def run_pipeline(config_path: str, verbose: bool = True):
    """Load and execute the pipeline functions."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data = None
    
    for step_config in config['pipeline']:
        func = load_function(
            step_config['module'],
            step_config['function']
        )
        
        params = step_config.get('params', {})
        
        if step_config['function'] == 'train_models':
            if 'models_config' in params:
                models_key = params.pop('models_config')
                params['models_config'] = config[models_key]

        if data is None:
            data = func(verbose=verbose, **params)
        else:
            data = func(data, verbose=verbose, **params)
    
    print("Pipeline complete.")
    print()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --config
    parser.add_argument('--config', required=True, 
        help="Specify the configuration file")

    # --quiet
    parser.add_argument('--quiet', action='store_true',
        help="Run pipeline without verbose output")
    
    args = parser.parse_args()

    verbose = not args.quiet
    final_data = run_pipeline(args.config, verbose=verbose)
