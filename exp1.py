"""Compatibility entry point for the gated positive control."""
import sys
from cpsat_experiments import main

if __name__ == "__main__":
    sys.argv.insert(1, "exp1")
    main()
