"""CLI for simd-kernels."""
import argparse, json, logging, os, sys
from typing import List, Optional

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="simd-kernels")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="Check SIMD support")
    b = sub.add_parser("benchmark", help="Run benchmarks")
    b.add_argument("--output", type=str, default="results")
    args = parser.parse_args(argv)

    if args.command is None:
        print("Usage: simd-kernels {check|benchmark}")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if args.command == "check":
        from simd_kernels.bindings import is_available, get_backend_name
        print(f"Backend: {get_backend_name()}")
        print(f"Available: {'YES' if is_available() else 'NO (run make)'}")
        return 0

    elif args.command == "benchmark":
        from simd_kernels.benchmark import benchmark_binary_gemm, format_results
        results = benchmark_binary_gemm()
        print(format_results(results))
        os.makedirs(args.output, exist_ok=True)
        data = [{"name": r.name, "size": r.size, "pytorch_ms": r.pytorch_ms,
                 "simd_ms": r.simd_ms, "speedup": r.speedup, "backend": r.backend} for r in results]
        with open(os.path.join(args.output, "simd_benchmarks.json"), "w") as f:
            json.dump(data, f, indent=2)
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
