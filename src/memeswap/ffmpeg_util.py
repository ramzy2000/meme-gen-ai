"""Locate ffmpeg and ffprobe and run them."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _winget_bin(name: str) -> str | None:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    root = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not root.is_dir():
        return None
    matches = sorted(root.glob(f"Gyan.FFmpeg*/ffmpeg-*/bin/{name}.exe"))
    if not matches:
        return None
    return str(matches[-1])


def tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    winget = _winget_bin(name)
    if winget:
        return winget
    raise FFmpegError(
        f"{name} was not found on PATH. Install ffmpeg and ffprobe, and prefer an LGPL build. "
        "Encoding with libx264 commonly requires a build configured with --enable-gpl."
    )


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"command timed out: {' '.join(cmd)}") from exc


def version_line() -> str:
    proc = run([tool("ffmpeg"), "-version"])
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip() or "ffmpeg -version failed")
    return proc.stdout.splitlines()[0].strip()
