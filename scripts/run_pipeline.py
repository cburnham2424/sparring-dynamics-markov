"""Thin CLI entry point for the full estimation + simulation pipeline."""
import os
import sys

# Running this file directly (`python scripts/run_pipeline.py`) only
# puts scripts/ on sys.path, not the repo root, so sparring_dynamics
# wouldn't otherwise be importable. Insert the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparring_dynamics.pipeline import main

if __name__ == "__main__":
    main()
