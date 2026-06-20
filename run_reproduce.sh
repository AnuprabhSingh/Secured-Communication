#!/bin/sh
cd "$(dirname "$0")"
PYTHON=${PYTHON:-/opt/homebrew/opt/python@3.14/bin/python3.14}

# Create symlink if needed
if [ ! -L .venv/bin/python ]; then
    ln -sf "$PYTHON" .venv/bin/python3.14
    ln -sf python3.14 .venv/bin/python3
    ln -sf python3 .venv/bin/python
fi

echo "Python: $($PYTHON --version)"
echo "Starting full paper reproduction..."
PYTHONPATH=src $PYTHON scripts/run_joint_ao_results.py

echo "Generating IEEE plots..."
PYTHONPATH=src $PYTHON scripts/generate_joint_ao_plots.py

echo "✓ Reproduction complete! Results in outputs_joint_ao/"
