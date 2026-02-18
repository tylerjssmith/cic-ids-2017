#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./finalize_model.sh [OPTIONS] --results RESULTS_FILE --model MODEL_NAME --data DATA_DIR --output OUTPUT_FILE

Train final model on full dataset and save to file.

OPTIONS:
    -h, --help              Show this help message and exit
    --quiet                 Run without verbose output

ARGUMENTS:
    --results RESULTS_FILE  Path to results file (e.g. models/results.pkl)
    --model MODEL_NAME      Name of model to finalize (e.g. xgboost_1)
    --data DATA_DIR         Path to directory containing data splits
    --output OUTPUT_FILE    Path to save finalized model (e.g. models/final.skops)

EXAMPLES:
    ./finalize_model.sh --results models/results.pkl --model xgboost_1 --data data/processed/all_multi --output models/final.skops
    ./finalize_model.sh --results models/results.pkl --model xgboost_1 --data data/processed/all_multi --output models/final.skops --quiet

EOF
}

RESULTS=""
MODEL=""
DATA=""
OUTPUT=""
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
        --model)
            MODEL=$2
            shift 2
            ;;
        --data)
            DATA=$2
            shift 2
            ;;
        --output)
            OUTPUT=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./finalize_model.sh --results RESULTS_FILE --model MODEL_NAME --data DATA_DIR --output OUTPUT_FILE [--quiet]"
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

if [ -z "$MODEL" ]; then
    echo "Error: --model argument is required"
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

if [ -z "$OUTPUT" ]; then
    echo "Error: --output argument is required"
    exit 1
fi

echo "Finalizing '$MODEL' in $RESULTS, using data in: $DATA"
python src/finalizing.py --results "$RESULTS" --model "$MODEL" --data "$DATA" --output "$OUTPUT" $QUIET