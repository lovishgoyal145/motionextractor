#!/usr/bin/env python3
"""
Downloads DWPose whole-body ONNX model weights directly from the official GitHub Release.
Can be executed on any device to prepare the local models directory:
    python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "dwpose"

MODELS = {
    "dw-ll_ucoco_384.onnx": {
        "url": "https://github.com/lovishgoyal145/motionextractor/releases/download/v1.0.0/dw-ll_ucoco_384.onnx",
        "fallback": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        "size_mb": 128.2,
    },
    "yolox_l.onnx": {
        "url": "https://github.com/lovishgoyal145/motionextractor/releases/download/v1.0.0/yolox_l.onnx",
        "fallback": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        "size_mb": 206.7,
    },
}


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    downloaded_mb = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent:3d}% [{downloaded_mb:.1f} MB / {total_mb:.1f} MB]")
    sys.stdout.flush()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("    DWPose ONNX Model Weights Downloader")
    print(f"    Target Directory: {MODELS_DIR}")
    print("=" * 60)

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[EXISTS] {filename} ({dest.stat().st_size / (1024*1024):.1f} MB)")
            continue

        print(f"\n[FETCH] {filename} (~{info['size_mb']} MB)...")
        try:
            urllib.request.urlretrieve(info["url"], str(dest), reporthook=download_progress)
            print(f"\n[DONE] Successfully downloaded {filename} from GitHub Release!")
        except Exception as e:
            print(f"\n[WARNING] GitHub Release download failed ({e}), trying HuggingFace fallback...")
            urllib.request.urlretrieve(info["fallback"], str(dest), reporthook=download_progress)
            print(f"\n[DONE] Successfully downloaded {filename} from fallback!")

    print("\nAll model weights are ready to use!")


if __name__ == "__main__":
    main()
