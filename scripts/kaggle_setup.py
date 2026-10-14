"""Kaggle environment setup.

Sets CHROMA_PATH and CACHE_DIR environment variables so the ingestion
pipeline and eval scripts write to Kaggle's fast temp storage rather
than the local defaults.

Run this ONCE at the top of a Kaggle notebook, before any pipeline or
eval invocation:

    python scripts/kaggle_setup.py

The variables are set in the current process. To persist them for
subsequent cells in a Kaggle notebook, run this script inside a cell
that does NOT spawn a subprocess, OR use the shell-injection pattern:

    import os
    os.environ["CHROMA_PATH"] = "/kaggle/tmp/chroma"
    os.environ["CACHE_DIR"] = "/kaggle/tmp/processed"

The env-var approach is used by store.py (CHROMA_PATH) and loader.py
(CACHE_DIR). Both fall back to local defaults when the env vars are
not set.
"""

from __future__ import annotations

import os
from pathlib import Path


KAGGLE_CHROMA_PATH = "/kaggle/tmp/chroma"
KAGGLE_CACHE_DIR = "/kaggle/tmp/processed"


def setup_kaggle_paths() -> None:
    """Set env vars and create directories for the Kaggle session."""
    os.environ["CHROMA_PATH"] = KAGGLE_CHROMA_PATH
    os.environ["CACHE_DIR"] = KAGGLE_CACHE_DIR

    Path(KAGGLE_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(KAGGLE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    print("Kaggle paths configured:")
    print(f"  CHROMA_PATH = {os.environ['CHROMA_PATH']}")
    print(f"  CACHE_DIR   = {os.environ['CACHE_DIR']}")
    print()
    print("Directories created (if missing):")
    print(f"  {KAGGLE_CHROMA_PATH}")
    print(f"  {KAGGLE_CACHE_DIR}")


def print_diagnostic() -> None:
    """Report what is set, for sanity-checking in the notebook."""
    print()
    print("Current environment state:")
    print(f"  CHROMA_PATH = {os.environ.get('CHROMA_PATH', 'NOT SET')}")
    print(f"  CACHE_DIR   = {os.environ.get('CACHE_DIR', 'NOT SET')}")
    print()
    print("To use these variables in later notebook cells, run the")
    print("following BEFORE importing src modules:")
    print()
    print('    import os')
    print(f'    os.environ["CHROMA_PATH"] = "{KAGGLE_CHROMA_PATH}"')
    print(f'    os.environ["CACHE_DIR"] = "{KAGGLE_CACHE_DIR}"')
    print()


if __name__ == "__main__":
    setup_kaggle_paths()
    print_diagnostic()