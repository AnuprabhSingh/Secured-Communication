#!/usr/bin/env python3
"""Create .venv/bin/python symlink and run reproduction."""
import os
import subprocess
import sys

venv_bin = os.path.join(os.path.dirname(__file__), ".venv", "bin")
python_target = "/opt/homebrew/opt/python@3.14/bin/python3.14"
python_link = os.path.join(venv_bin, "python")

# Create symlink if needed
if not os.path.exists(python_link):
    try:
        os.symlink(python_target, python_link)
        print(f"Created symlink: {python_link} -> {python_target}")
    except OSError as e:
        print(f"Failed to create symlink: {e}")
        print(f"Please run: ln -sf {python_target} {python_link}")
        sys.exit(1)

# Run reproduction
script_dir = os.path.dirname(__file__)
scripts_dir = os.path.join(script_dir, "scripts")

print("Running full paper reproduction (joint_ao)...")
rc = subprocess.call([
    python_link,
    os.path.join(scripts_dir, "run_joint_ao_results.py")
])
if rc != 0:
    print(f"Paper reproduction failed with exit code {rc}")
    sys.exit(rc)

print("\nGenerating IEEE plots...")
rc = subprocess.call([
    python_link,
    os.path.join(scripts_dir, "generate_joint_ao_plots.py")
])
if rc != 0:
    print(f"Plot generation failed with exit code {rc}")
    sys.exit(rc)

print("\n✓ Reproduction complete! Results in outputs_joint_ao/")
sys.exit(0)
