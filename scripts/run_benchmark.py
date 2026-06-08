#!/usr/bin/env python
"""Run SIMD kernel benchmarks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simd_kernels.cli import main
if __name__ == "__main__": sys.exit(main(["benchmark"] + sys.argv[1:]))
