"""Call the pinned GHOST inference.py entrypoint. Do not invent flags it does not have."""

from __future__ import annotations

import subprocess
from pathlib import Path


class SwapError(RuntimeError):
    pass


def _tail(text: str | None, limit: int = 20000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def video_command(
    python: str,
    sources: list[Path],
    target_faces: list[Path],
    video: Path,
    out_video: Path,
) -> list[str]:
    return [
        python,
        "inference.py",
        "--source_paths",
        *[str(path) for path in sources],
        "--target_faces_paths",
        *[str(path) for path in target_faces],
        "--target_video",
        str(video),
        "--out_video_name",
        str(out_video),
    ]


def image_command(python: str, sources: list[Path], target_image: Path, out_image: Path) -> list[str]:
    # argparse in inference.py uses type=bool, so the string "True" is required.
    # Any non-empty string would also become True; do not pass "False".
    return [
        python,
        "inference.py",
        "--source_paths",
        *[str(path) for path in sources],
        "--target_image",
        str(target_image),
        "--image_to_image",
        "True",
        "--out_image_name",
        str(out_image),
    ]


def run_ghost(cmd: list[str], cwd: Path, timeout: float | None) -> None:
    cwd = Path(cwd)
    if not (cwd / "inference.py").is_file():
        raise SwapError(
            f"GHOST inference.py is not at {cwd}. "
            "Run: git submodule update --init --recursive"
        )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        raise SwapError(f"GHOST timed out after {timeout}s\n{_tail(stderr)}") from exc
    if proc.returncode != 0:
        raise SwapError(
            f"GHOST exited {proc.returncode}\n{_tail(proc.stderr)}\n{_tail(proc.stdout)}"
        )
