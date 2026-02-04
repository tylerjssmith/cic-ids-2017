#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: ./run_system.sh [OPTIONS] [CONFIG_FILE]

Run system to train, test, or finalize model using a configuration file. The
file usually will end in _train.yml, _test.yml, or final.yml. Configuration
files can be used to specify data processing or models.

OPTIONS:
    -h, --help      Show this help message and exit
    --quiet         Run system without verbose output

ARGUMENTS:
    CONFIG_FILE     Path to YAML configuration file

EXAMPLES:
    ./run_system.sh config.yaml            
    ./run_system.sh config.yaml --quiet    # Run quietly
    ./run_system.sh --quiet config.yaml    # Arguments in any order

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
            echo "Usage: ./run_system.sh [config.yml] [--quiet]"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

echo "Running system with config: $CONFIG"
python src/main.py --config "$CONFIG" $QUIET
