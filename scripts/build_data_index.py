# Data index builder.
# Author: Haoyu Qiang.

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cache_manager import CacheManager
from src.data.dataset_registry import get_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a JSON index of cached dataset files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()

    entry = get_entry(args.dataset)
    cache = CacheManager(args.cache_root)
    dataset_dir = cache.dataset_dir(entry.name)
    files = sorted(str(path.relative_to(dataset_dir)) for path in dataset_dir.rglob("*") if path.is_file())
    payload = cache.read_index()
    payload[entry.name] = {"files": files, "root": str(dataset_dir)}
    cache.write_index(payload)
    print(f"Indexed {len(files)} files for {entry.name}.")


if __name__ == "__main__":
    main()
