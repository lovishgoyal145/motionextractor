#!/usr/bin/env python3
"""
Publishes DWPose ONNX model weights to GitHub Releases for lovishgoyal145/motionextractor.
GitHub blocks files > 100MB in standard git pushes, so Releases are the industry standard
for distributing large ML weights (>100MB up to 2GB per file).
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_github_pat() -> str:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip().startswith("GITHUB_PAT="):
                    return line.strip().split("=", 1)[1].strip().strip("'\"")
    pat = os.getenv("GITHUB_PAT", "")
    if not pat:
        raise ValueError("GITHUB_PAT not found in .env or environment")
    return pat


def create_or_get_release(owner: str, repo: str, tag: str, pat: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "MotionExtractor-Publisher",
    }

    # Check existing releases
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            releases = json.loads(resp.read().decode())
            for r in releases:
                if r.get("tag_name") == tag:
                    print(f"Found existing release: {tag} (id: {r['id']})")
                    return r
    except Exception as e:
        print(f"Checking existing releases: {e}")

    # Create new release
    print(f"Creating new GitHub Release {tag}...")
    data = json.dumps({
        "tag_name": tag,
        "name": f"{tag} - DWPose Model Weights (133 Keypoints)",
        "body": "Official ONNX model weights for DWPose whole-body 133-keypoint motion extraction and YOLOX-L multi-person detection.\n\nFiles:\n- `dw-ll_ucoco_384.onnx` (~129MB)\n- `yolox_l.onnx` (~207MB)",
        "draft": False,
        "prerelease": False,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"Created release {tag} successfully (id: {res['id']})")
        return res


def upload_asset(upload_url_template: str, file_path: Path, pat: str):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    # URL template is like: https://uploads.github.com/repos/owner/repo/releases/ID/assets{?name,label}
    upload_url = upload_url_template.split("{")[0] + f"?name={file_path.name}"
    file_size = file_path.stat().st_size
    print(f"\nUploading {file_path.name} ({file_size / (1024*1024):.1f} MB) to GitHub Releases...")

    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(file_size),
        "User-Agent": "MotionExtractor-Publisher",
    }

    t0 = time.time()
    with open(file_path, "rb") as f:
        req = urllib.request.Request(upload_url, data=f, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            elapsed = time.time() - t0
            print(f"Successfully uploaded {file_path.name} in {elapsed:.1f}s ({file_size / (1024*1024*elapsed):.2f} MB/s)!")
            print(f"Download URL: {data.get('browser_download_url')}")
            return data


def main():
    owner = "lovishgoyal145"
    repo = "motionextractor"
    tag = "v1.0.0"

    pat = get_github_pat()
    release = create_or_get_release(owner, repo, tag, pat)
    upload_url = release["upload_url"]

    existing_assets = {a["name"]: a for a in release.get("assets", [])}

    models_dir = REPO_ROOT / "models" / "dwpose"
    for fname in ["dw-ll_ucoco_384.onnx", "yolox_l.onnx"]:
        fpath = models_dir / fname
        if fname in existing_assets:
            print(f"Asset '{fname}' is already uploaded: {existing_assets[fname].get('browser_download_url')}")
        else:
            upload_asset(upload_url, fpath, pat)

    print("\nAll model weights are published and accessible globally!")


if __name__ == "__main__":
    main()
