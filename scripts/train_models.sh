#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./train_models.sh [OPTIONS] [CONFIG_FILE]

Run system to train machine learning models using a configuration file.

OPTIONS:
    -h, --help      Show this help message and exit
    --quiet         Run system without verbose output

ARGUMENTS:
    CONFIG_FILE     Path to YAML configuration file

EXAMPLES:
    ./train_models.sh config.yaml            
    ./train_models.sh config.yaml --quiet    # Run quietly
    ./train_models.sh --quiet config.yaml    # Arguments in any order

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
            echo "Usage: ./train_models.sh [config.yml] [--quiet]"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Running training pipeline with config: $CONFIG"
python src/main.py --config "$CONFIG" $QUIET
