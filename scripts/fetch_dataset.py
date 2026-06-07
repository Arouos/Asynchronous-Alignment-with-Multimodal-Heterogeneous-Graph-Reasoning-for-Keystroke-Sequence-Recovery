# Dataset download script.
# Author: Haoyu Qiang.

from __future__ import annotations

import argparse

from src.data.cache_manager import CacheManager
from src.data.dataset_registry import get_entry
from src.data.github_downloader import stream_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a registered dataset from its GitHub mirror.")
    parser.add_argument("--dataset", required=True, help="Dataset name as registered in src/data/dataset_registry.py")
    parser.add_argument("--cache-root", default=None, help="Optional override for the cache root.")
    args = parser.parse_args()

    entry = get_entry(args.dataset)
    cache = CacheManager(args.cache_root)
    archive_path = cache.archive_path(entry.name, entry.archive_format)
    stream_download(entry.github_url, archive_path)
    cache.extract_archive(archive_path, cache.dataset_dir(entry.name))


if __name__ == "__main__":
    main()
