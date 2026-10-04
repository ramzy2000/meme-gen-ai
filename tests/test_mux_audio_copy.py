from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from memeswap.ffmpeg_util import tool
from memeswap.ingest import probe
from memeswap.mux import build_mux_command, mux


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)


def _pcm_md5(path: Path) -> str:
    proc = subprocess.run(
        [
            tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode("utf-8", "replace"))
    return hashlib.md5(proc.stdout).hexdigest()


def test_build_command_copies_audio(tmp_path: Path) -> None:
    cmd = build_mux_command(
        ffmpeg="ffmpeg",
        swapped_video=tmp_path / "swapped.mp4",
        original_video=tmp_path / "in.mp4",
        output=tmp_path / "out.mp4",
        crf=18,
        fps_rational="30/1",
        duration=1.0,
        copy_audio=True,
    )
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "aac" not in cmd
    assert "libx264" in cmd


def test_mux_copies_audio_stream(tmp_path: Path) -> None:
    original = tmp_path / "original.mp4"
    swapped = tmp_path / "swapped.mp4"
    output = tmp_path / "out.mp4"
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=30:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(original),
        ]
    )
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=160x120:r=30:d=1",
            "-an",
            "-c:v",
            "mpeg4",
            str(swapped),
        ]
    )
    info = probe(original)
    before = _pcm_md5(original)
    out_info = mux(
        swapped,
        original,
        output,
        crf=18,
        fps=info["fps"],
        fps_rational=info["fps_rational"],
        duration=info["duration"],
        copy_audio=True,
    )
    assert out_info["audio_codec"] == info["audio_codec"]
    assert _pcm_md5(output) == before
