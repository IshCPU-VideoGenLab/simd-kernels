#!/usr/bin/env python
"""Detect SIMD capabilities."""
import sys, os, platform
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Processor: {platform.processor()}")
try:
    from simd_kernels.bindings import is_available, get_backend_name
    print(f"Backend: {get_backend_name()}")
    print(f"Available: {'YES' if is_available() else 'NO (run make)'}")
except Exception as e:
    print(f"Library not loaded: {e}")
