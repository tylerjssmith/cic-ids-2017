#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./run_pipeline.sh [OPTIONS] [CONFIG_FILE]

Run the training pipeline with the specified configuration file.

OPTIONS:
    -h, --help      Show this help message and exit
    --quiet         Run pipeline without verbose output

ARGUMENTS:
    CONFIG_FILE     Path to YAML configuration file

EXAMPLES:
    ./run_pipeline.sh config.yaml            
    ./run_pipeline.sh config.yaml --quiet    # Run quietly
    ./run_pipeline.sh --quiet config.yaml    # Arguments in any order

EOF
}

CONFIG=""
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
        *.yaml|*.yml)
            CONFIG=$1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./run_pipeline.sh [config.yml] [--quiet]"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Running pipeline with config: $CONFIG"
python src/main.py --config "$CONFIG" $QUIET
