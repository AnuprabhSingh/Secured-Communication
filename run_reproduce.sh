#!/bin/sh
cd "/Users/anuprabh/Desktop/Secured Comm"
PYTHON=/opt/homebrew/opt/python@3.14/bin/python3.14

# Create symlink if needed
if [ ! -L .venv/bin/python ]; then
    ln -sf "$PYTHON" .venv/bin/python3.14
    ln -sf python3.14 .venv/bin/python3
    ln -sf python3 .venv/bin/python
fi

echo "Python: $($PYTHON --version)"
echo "Starting reproduce profile..."
PYTHONPATH=src $PYTHON scripts/run_paper_trends.py --profile reproduce --output outputs_reproduce_v3
echo "Running validation..."
PYTHONPATH=src $PYTHON scripts/check_scientific_reproduction.py --results outputs_reproduce_v3/results.json
echo "Done."
