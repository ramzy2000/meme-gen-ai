"""Probe a target clip and reject inputs ffmpeg cannot treat as constant-frame-rate video."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from memeswap.ffmpeg_util import FFmpegError, run, tool

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


class IngestError(RuntimeError):
    pass


def _fraction(rate: str) -> Fraction:
    if not rate or rate in {"0/0", "N/A"}:
        raise IngestError(f"cannot prove a constant frame rate from rate {rate!r}")
    if "/" not in rate:
        raise IngestError(f"ffprobe frame rate is not a rational: {rate!r}")
    num_s, den_s = rate.split("/", 1)
    try:
        num, den = int(num_s), int(den_s)
    except ValueError as exc:
        raise IngestError(f"ffprobe frame rate is not a rational: {rate!r}") from exc
    if den == 0:
        raise IngestError(f"cannot prove a constant frame rate from rate {rate!r}")
    return Fraction(num, den)


def probe(path: Path) -> dict:
    """Return fps, duration, audio codec, and size for a decodable CFR clip."""
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"unsupported extension {path.suffix!r}; accepted extensions are mp4, mov, and webm"
        )
    if not path.is_file():
        raise IngestError(f"target does not exist: {path}")

    proc = run(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "ffprobe failed"
        raise IngestError(f"ffprobe rejected {path}: {detail}")
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"ffprobe did not return JSON for {path}") from exc

    video_streams = [s for s in raw.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise IngestError(f"{path} has no video stream")
    video = video_streams[0]
    audio_streams = [s for s in raw.get("streams", []) if s.get("codec_type") == "audio"]
    audio_codec = audio_streams[0].get("codec_name") if audio_streams else None

    decode = run(
        [
            tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    if decode.returncode != 0:
        detail = decode.stderr.strip() or "ffmpeg could not decode a frame"
        raise IngestError(f"ffmpeg cannot decode {path}: {detail}")

    r_rate = video.get("r_frame_rate")
    avg_rate = video.get("avg_frame_rate")
    r_frac = _fraction(str(r_rate))
    avg_frac = _fraction(str(avg_rate))
    if r_frac != avg_frac:
        raise IngestError(
            "variable frame rate that cannot be proved constant: "
            f"r_frame_rate={r_rate} avg_frame_rate={avg_rate}"
        )

    fmt = raw.get("format") or {}
    try:
        duration = float(fmt["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(f"ffprobe did not report a duration for {path}") from exc
    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(f"ffprobe did not report a frame size for {path}") from exc

    fps = float(r_frac)
    if fps <= 0:
        raise IngestError(f"ffprobe reported a non-positive frame rate for {path}")

    return {
        "path": str(path.resolve()),
        "fps": fps,
        "fps_rational": f"{r_frac.numerator}/{r_frac.denominator}",
        "duration": duration,
        "audio_codec": audio_codec,
        "width": width,
        "height": height,
        "video_codec": video.get("codec_name"),
        "format_name": fmt.get("format_name"),
        "probe": raw,
    }


def frame_count(info: dict) -> int:
    """Best-effort frame count from duration and the constant frame rate."""
    return max(1, int(round(info["duration"] * info["fps"])))
