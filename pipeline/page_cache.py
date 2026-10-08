"""Release the page cache held by large read-only files before the reader server allocates GPU memory.

    python -m pipeline.page_cache [path ...]        (default: the model cache and the source videos)

On a unified-memory GPU (DGX Spark, GB10) CUDA allocations do not reclaim the page cache, so the server can fail with
"out of memory" while most memory is clean cache. posix_fadvise(DONTNEED) on model weights and videos, which can be
re-read from disk, frees that memory without root privileges and without affecting any process; dirty pages are kept.
"""
import os
import sys

from .common import SHARED

MIN_BYTES = 50_000_000  # smaller files hold too little cache to matter


def release(paths: list) -> int:
    total = 0
    for root in paths:
        for dirpath, _, names in os.walk(root):
            for name in names:
                path = os.path.join(dirpath, name)
                try:
                    if os.path.islink(path) or (size := os.path.getsize(path)) < MIN_BYTES:
                        continue
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                    total += size
                except OSError:
                    continue
    return total


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(SHARED / "hf-cache" / "hub"), str(SHARED / "raw")]
    print(f"page cache released for {release(paths) / 1e9:.1f} GB of files")
