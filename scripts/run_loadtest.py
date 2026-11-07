"""Reproducible isolated real-stack concurrency/stability suite."""

import sys

from ai4edu.loadtest import main

if __name__ == "__main__":
    sys.argv.insert(1, "run")
    main()
