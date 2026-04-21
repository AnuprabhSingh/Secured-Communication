#!/usr/bin/env python3
"""Bootstrap: create .venv/bin/python symlink, then exec the real script."""
import os, sys

venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
target = "/opt/homebrew/opt/python@3.14/bin/python3.14"

if not os.path.exists(venv_python):
    os.symlink(target, venv_python)
    print(f"Created {venv_python} -> {target}")

# Now exec the reproduce script using the venv python
os.execv(venv_python, [venv_python, "scripts/run_paper_trends.py", "--profile", "reproduce", "--output", "outputs_reproduce_v3"])
