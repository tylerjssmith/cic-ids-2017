#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./evaluate_models.sh [OPTIONS] --results RESULTS_FILE --data DATA_DIR

Evaluate trained machine learning models using test data.

OPTIONS:
    -h, --help              Show this help message and exit
    --quiet                 Run without verbose output

ARGUMENTS:
    --results RESULTS_FILE  Path to results file (e.g. models/results.pkl)
    --data DATA_DIR         Path to directory containing data splits

EXAMPLES:
    ./evaluate_models.sh --results models/results.pkl --data data/processed/
    ./evaluate_models.sh --results models/results.pkl --data data/processed/ --quiet

EOF
}

RESULTS=""
DATA=""
QUIET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --quiet)
            QUIET="--quiet"
            shift
            ;;
        --results)
            RESULTS=$2
            shift 2
            ;;
        --data)
            DATA=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./evaluate_models.sh --results RESULTS_FILE --data DATA_DIR [--quiet]"
            exit 1
            ;;
    esac
done

if [ -z "$RESULTS" ]; then
    echo "Error: --results argument is required"
    exit 1
fi

if [ ! -f "$RESULTS" ]; then
    echo "Error: Results file '$RESULTS' not found"
    exit 1
fi

if [ -z "$DATA" ]; then
    echo "Error: --data argument is required"
    exit 1
fi

if [ ! -d "$DATA" ]; then
    echo "Error: Data directory '$DATA' not found"
    exit 1
fi

echo "Running evaluation pipeline with results: $RESULTS and data: $DATA"
python src/testing.py --results "$RESULTS" --data "$DATA" $QUIET
