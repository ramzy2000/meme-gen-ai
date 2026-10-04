"""Put a winget ffmpeg install on PATH when it is not already there."""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure() -> None:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return
    root = Path(local) / "Microsoft" / "WinGet" / "Packages"
    bins = sorted(root.glob("Gyan.FFmpeg*/ffmpeg-*/bin"))
    if not bins:
        return
    os.environ["PATH"] = str(bins[-1]) + os.pathsep + os.environ.get("PATH", "")
