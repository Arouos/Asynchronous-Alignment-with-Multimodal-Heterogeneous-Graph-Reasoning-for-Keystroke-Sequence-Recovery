# Environment preparation script.
# Author: Haoyu Qiang.

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REQUIRED = (
    "torch",
    "torchaudio",
    "numpy",
    "scipy",
    "sklearn",
    "librosa",
    "yaml",
    "tqdm",
    "requests",
    "einops",
)


def main() -> int:
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required.", file=sys.stderr)
        return 1
    missing: list[str] = []
    for package in REQUIRED:
        try:
            import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        return 2
    from src.utils.path_utils import default_cache_root

    cache_root = default_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Cache root ready at {cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
