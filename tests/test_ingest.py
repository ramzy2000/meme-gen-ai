from __future__ import annotations

from pathlib import Path

import pytest

from memeswap.detect import Face, frame_is_ambiguous
from memeswap.ffmpeg_util import tool
from memeswap.ingest import IngestError, probe


def _run(cmd: list[str]) -> None:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)


def _good_mp4(path: Path) -> None:
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
            str(path),
        ]
    )


def test_probe_constant_mp4(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _good_mp4(video)
    info = probe(video)
    assert info["fps"] == 30
    assert info["width"] == 160
    assert info["height"] == 120
    assert info["audio_codec"] == "aac"
    assert info["video_codec"] == "h264"
    assert info["duration"] > 0


def test_reject_bad_extension(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    path.write_bytes(b"not a video")
    with pytest.raises(IngestError, match="unsupported extension"):
        probe(path)


def test_reject_missing_video_stream(tmp_path: Path) -> None:
    path = tmp_path / "audio.mp4"
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            "aac",
            str(path),
        ]
    )
    with pytest.raises(IngestError, match="no video stream"):
        probe(path)


def test_reject_undecodable(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not an mp4")
    with pytest.raises(IngestError):
        probe(path)


def test_reject_variable_frame_rate(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    listing = tmp_path / "list.txt"
    mixed = tmp_path / "vfr.mp4"
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=30:d=1",
            "-c:v",
            "libx264",
            str(first),
        ]
    )
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=24:d=1",
            "-c:v",
            "libx264",
            str(second),
        ]
    )
    listing.write_text(f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n", encoding="utf-8")
    _run(
        [
            tool("ffmpeg"),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(mixed),
        ]
    )
    with pytest.raises(IngestError, match="variable frame rate"):
        probe(mixed)


def test_similar_faces_are_ambiguous() -> None:
    faces = [Face(0, 0, 100, 100, 0.9), Face(10, 10, 90, 90, 0.8)]
    assert frame_is_ambiguous(faces, 0.75)
    distinct = [Face(0, 0, 100, 100, 0.9), Face(10, 10, 20, 20, 0.8)]
    assert not frame_is_ambiguous(distinct, 0.75)
