"""Final H.264 encode. Audio is always stream-copied from the original clip."""

from __future__ import annotations

from pathlib import Path

from memeswap.ffmpeg_util import run, tool
from memeswap.ingest import probe


class MuxError(RuntimeError):
    pass


def build_mux_command(
    *,
    ffmpeg: str,
    swapped_video: Path,
    original_video: Path,
    output: Path,
    crf: int,
    fps_rational: str,
    duration: float,
    copy_audio: bool,
) -> list[str]:
    cmd = [ffmpeg, "-y", "-i", str(swapped_video)]
    if copy_audio:
        cmd += ["-i", str(original_video), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-r",
        fps_rational,
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{duration:.6f}",
        str(output),
    ]
    return cmd


def mux(
    swapped_video: Path,
    original_video: Path,
    output: Path,
    *,
    crf: int,
    fps: float,
    fps_rational: str,
    duration: float,
    copy_audio: bool,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_mux_command(
        ffmpeg=tool("ffmpeg"),
        swapped_video=Path(swapped_video),
        original_video=Path(original_video),
        output=output,
        crf=crf,
        fps_rational=fps_rational,
        duration=duration,
        copy_audio=copy_audio,
    )
    proc = run(cmd)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "ffmpeg failed"
        raise MuxError(detail)
    if not output.is_file():
        raise MuxError(f"ffmpeg did not write {output}")
    info = probe(output)
    limit = 1.0 / fps
    delta = abs(info["duration"] - duration)
    if delta > limit:
        raise MuxError(
            f"output duration {info['duration']:.6f}s differs from input {duration:.6f}s "
            f"by more than one frame ({limit:.6f}s)"
        )
    return info
