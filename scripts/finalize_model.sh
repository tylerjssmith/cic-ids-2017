#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./finalize_model.sh [OPTIONS] [CONFIG_FILE]

Finalize a network intrusion detection model with the specified configuration file.

OPTIONS:
    -h, --help      Show this help message and exit
    --quiet         Run script without verbose output

ARGUMENTS:
    CONFIG_FILE     Path to YAML configuration file

EXAMPLES:
    ./finalize_model.sh config.yaml            
    ./finalize_model.sh config.yaml --quiet    # Run quietly
    ./finalize_model.sh --quiet config.yaml    # Arguments in any order

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
            echo "Usage: ./finalize_model.sh [config.yml] [--quiet]"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Finalizing model with config: $CONFIG"
python src/main.py --config "$CONFIG" $QUIET
