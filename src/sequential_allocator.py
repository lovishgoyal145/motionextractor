"""
Sequential Output Directory Allocator
Ensures safe, monotonic directory allocation under /home/lovish/motion-data/
e.g. 0001, 0002, 0003...
Guaranteed never to overwrite an existing extraction.
"""

import os
from pathlib import Path
from typing import Tuple


DEFAULT_OUTPUT_BASE = Path("/home/lovish/motion-data")


def get_next_sequence_dir(base_dir: Path = DEFAULT_OUTPUT_BASE) -> Tuple[Path, str]:
    """
    Safely finds the next sequential directory under base_dir (e.g. 0001, 0002).
    Creates the directory atomically with exist_ok=False to avoid race conditions.
    Returns (out_dir_path, sequence_id_str).
    """
    base_dir = Path(base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find all existing subdirectories matching exactly digits
    existing_nums = []
    if base_dir.exists():
        for entry in base_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                try:
                    existing_nums.append(int(entry.name))
                except ValueError:
                    continue

    next_num = max(existing_nums, default=0) + 1

    # Atomic creation loop in case another process allocated in the meantime
    while True:
        seq_str = f"{next_num:04d}"
        candidate_dir = base_dir / seq_str
        try:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            return candidate_dir, seq_str
        except FileExistsError:
            next_num += 1
