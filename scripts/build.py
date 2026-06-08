#!/usr/bin/env python
"""Build the SIMD kernel library."""
import subprocess, os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run(["make", "-C", root], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0: print(r.stderr); sys.exit(1)
subprocess.run(["make", "-C", root, "install"], capture_output=True)
print("Build complete.")
