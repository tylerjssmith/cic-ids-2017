"""
Run machine learning pipelines for network intrusion detection.
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
    print(f'Running pipeline with {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    obj = None
    
    for step in config['pipeline']:
        func = load_function(
            step['module'],
            step['function']
        )
        
        params = step.get('params', {}).copy()
        
        if 'models' in params and params['models'] == 'models':
            params['models'] = config['models']

        if obj is None:
            obj = func(verbose=verbose, **params)
        else:
            obj = func(obj, verbose=verbose, **params)
    
    print('Pipeline complete')
    print()

    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --config
    parser.add_argument('--config', required=True, 
        help='Specify the configuration file')

    # --quiet
    parser.add_argument('--quiet', action='store_true',
        help='Run pipeline without verbose output')
    
    args = parser.parse_args()

    verbose = not args.quiet
    final_obj = run_pipeline(args.config, verbose=verbose)