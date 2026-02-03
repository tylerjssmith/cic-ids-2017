#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./test_models.sh [OPTIONS] [CONFIG_FILE]

Run the testing pipeline with the specified configuration file.

OPTIONS:
    -h, --help      Show this help message and exit
    --quiet         Run pipeline without verbose output

ARGUMENTS:
    CONFIG_FILE     Path to YAML configuration file

EXAMPLES:
    ./test_models.sh config.yaml            
    ./test_models.sh config.yaml --quiet    # Run quietly
    ./test_models.sh --quiet config.yaml    # Arguments in any order

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
            echo "Usage: ./test_models.sh [config.yml] [--quiet]"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Running testing pipeline with config: $CONFIG"
python src/main.py --config "$CONFIG" $QUIET
