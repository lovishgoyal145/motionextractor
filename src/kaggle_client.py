"""
Kaggle Client Mechanism copied and adapted from SwapeDev.
Provides isolated environment creation, credential management, kernel push,
execution monitoring, and output downloading.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_env_credentials(env_path: Optional[Path] = None) -> Tuple[str, str]:
    """Reads KAGGLE_USERNAME and KAGGLE_KEY from .env file or environment."""
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()

    if not username or not key:
        candidate_paths = [
            env_path,
            Path(__file__).resolve().parents[1] / ".env",
            Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/.env"),
            Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/swapedev_service/.env"),
        ]
        for cp in candidate_paths:
            if cp and cp.exists() and cp.is_file():
                with open(cp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k == "KAGGLE_USERNAME" and not username:
                            username = v
                        elif k == "KAGGLE_KEY" and not key:
                            key = v
                if username and key:
                    break

    return username, key


def create_isolated_kaggle_env(
    username: str,
    key: str,
    temp_dir: Path,
) -> Dict[str, str]:
    """
    Prepares an isolated Kaggle config directory and environment dict.
    Strictly isolated: does NOT mutate global os.environ.
    Replicates SwapeDev's create_isolated_kaggle_env.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = temp_dir / "kaggle.json"
    with open(cfg_file, "w", encoding="utf-8") as kf:
        json.dump({"username": username, "key": key}, kf)
    try:
        os.chmod(cfg_file, 0o600)
    except Exception:
        pass

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(temp_dir)
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    if key.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = key

    user_bin_dir = "/home/lovish/.local/bin"
    venv_bin_dir = str(Path(sys.executable).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = f"{venv_bin_dir}:{user_bin_dir}:{current_path}"
    return env


def find_kaggle_executable() -> list:
    """Locates the kaggle CLI command or python module runner."""
    user_bin = Path("/home/lovish/.local/bin/kaggle")
    if user_bin.exists() and os.access(user_bin, os.X_OK):
        return [str(user_bin)]
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin and os.path.isfile(kaggle_bin) and os.access(kaggle_bin, os.X_OK):
        return [kaggle_bin]
    venv_bin = Path(sys.executable).parent / "kaggle"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return [str(venv_bin)]
    return [sys.executable, "-m", "kaggle"]


class KagglePipelineRunner:
    """Manages Kaggle kernel push, status monitoring, and output downloading."""

    def __init__(self, username: Optional[str] = None, key: Optional[str] = None):
        u, k = load_env_credentials()
        self.username = username or u
        self.key = key or k
        if not self.username or not self.key:
            raise ValueError("Kaggle credentials not found in environment or .env file.")

        self.temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_motion_"))
        self.env = create_isolated_kaggle_env(self.username, self.key, self.temp_dir)
        self.cmd_prefix = find_kaggle_executable()

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def push_kernel(self, config_dir: Path) -> subprocess.CompletedProcess:
        """Pushes kernel from config_dir containing kernel-metadata.json."""
        cmd = self.cmd_prefix + ["kernels", "push", "-p", str(config_dir)]
        return subprocess.run(cmd, env=self.env, capture_output=True, text=True)

    def get_status(self, kernel_slug: str) -> str:
        """Queries status of kernel (e.g. RUNNING, COMPLETE, ERROR)."""
        cmd = self.cmd_prefix + ["kernels", "status", kernel_slug]
        res = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        return res.stdout.strip()

    def download_outputs(self, kernel_slug: str, output_dir: Path) -> subprocess.CompletedProcess:
        """Downloads kernel output files to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.cmd_prefix + ["kernels", "output", kernel_slug, "-p", str(output_dir)]
        return subprocess.run(cmd, env=self.env, capture_output=True, text=True)
