#!/usr/bin/env bash

# Parse command-line arguments
DEBUG_MODE=0  # Default to not start in debug mode
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1 ;;  # Enable debug mode if --debug flag is passed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Start the virtual environment
echo "Starting virtual environment"
source .venv/bin/activate

# Set environment variables
export FLASK_APP=app
export FLASK_ENV=development

# Conditionally set debug mode
if [[ $DEBUG_MODE -eq 1 ]]; then
    export FLASK_DEBUG=1
    echo "Debug mode enabled"
else
    export FLASK_DEBUG=0
    echo "Debug mode disabled"
fi

# Start Flask in the background
echo "Starting Flask with FLASK_APP=$FLASK_APP, FLASK_ENV=$FLASK_ENV, and FLASK_DEBUG=$FLASK_DEBUG..."
python -m flask run
