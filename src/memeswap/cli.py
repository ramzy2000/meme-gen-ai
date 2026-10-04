"""memeswap command line. Three commands, no web UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from memeswap import __version__
from memeswap.config import load_config, root_path
from memeswap.detect import DetectError, build_target_crop
from memeswap.enhance import EnhanceError, enhance_video
from memeswap.ffmpeg_util import run, tool, version_line
from memeswap.ingest import IngestError, probe
from memeswap.mux import MuxError, mux
from memeswap.swap import SwapError, image_command, run_ghost, video_command

GHOST_COMMIT = "44e58aad8600ee83ad4c8213aaa9698ccaf66c1c"
YUNET_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
GFPGAN_COMMIT = "7552a7791caad982045a7bbe5634bbf1cd5c8679"


def _job_id(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _timeout(config: dict) -> float | None:
    raw = config.get("timeout_seconds") or 0
    value = float(raw)
    return value if value > 0 else None


def _require_files(paths: list[Path], label: str) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def _split_chunks(video: Path, work: Path, seconds: float) -> list[Path]:
    if not seconds or seconds <= 0:
        return [video]
    out_dir = work / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%03d.mp4")
    proc = run(
        [
            tool("ffmpeg"),
            "-y",
            "-i",
            str(video),
            "-an",
            "-c:v",
            "libx264",
            "-f",
            "segment",
            "-segment_time",
            str(seconds),
            "-reset_timestamps",
            "1",
            pattern,
        ]
    )
    if proc.returncode != 0:
        raise MuxError(proc.stderr.strip() or "failed to split the clip into chunks")
    chunks = sorted(out_dir.glob("chunk_*.mp4"))
    if not chunks:
        raise MuxError("chunking produced no files")
    return chunks


def _concat(parts: list[Path], dest: Path) -> Path:
    if len(parts) == 1:
        return parts[0]
    listing = dest.with_suffix(".txt")
    lines = []
    for part in parts:
        escaped = part.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc = run(
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
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MuxError(proc.stderr.strip() or "failed to concatenate swapped chunks")
    return dest


def _gpu_report() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not available"
    proc = subprocess.run(
        [exe, "--query-gpu=name,memory.total", "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return text or "nvidia-smi failed"
    return text or "nvidia-smi returned no GPU rows"


def _sidecar(output: Path, payload: dict) -> Path:
    path = output.with_suffix(".json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _public_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def swap_image(sources: list[Path], target: Path, output: Path, config: dict) -> None:
    _require_files(sources, "source")
    _require_files([target], "target")
    output.parent.mkdir(parents=True, exist_ok=True)
    ghost = root_path(config, "ghost_dir", "third_party/ghost")
    cmd = image_command(str(config.get("ghost_python") or "python"), sources, target.resolve(), output.resolve())
    run_ghost(cmd, ghost, _timeout(config))
    if not output.is_file():
        raise SwapError(f"GHOST did not write {output}")


def swap_video(
    sources: list[Path],
    target: Path,
    output: Path,
    config: dict,
    *,
    crf: int,
    face_index: int | None,
    target_crops: list[Path],
    enhance: bool,
    consent: bool,
) -> dict:
    if not consent:
        raise IngestError("swap-video requires --consent")
    if face_index is not None and face_index < 0:
        raise DetectError("--face-index counts from 0, largest face first")
    _require_files(sources, "source")
    info = probe(target)
    work = root_path(config, "work_dir", "work") / _job_id(target)
    work.mkdir(parents=True, exist_ok=True)
    ghost = root_path(config, "ghost_dir", "third_party/ghost")
    python = str(config.get("ghost_python") or "python")

    if len(sources) > 1 and len(target_crops) != len(sources):
        raise DetectError(
            "more than one --source requires the same number of --target-crop stills, "
            "one per person, in the same order"
        )
    if len(sources) == 1 and len(target_crops) > 1:
        raise DetectError("one --source accepts at most one --target-crop")

    if len(sources) == 1 and not target_crops:
        crop = build_target_crop(
            target,
            info,
            work,
            model_path=root_path(config, "yunet_model", "models/face_detection_yunet_2023mar.onnx"),
            sample_stride=int(config.get("sample_stride") or 10),
            similar_area_ratio=float(config.get("similar_area_ratio") or 0.75),
            face_index=face_index,
            score_threshold=float(config.get("yunet_score_threshold") or 0.6),
            nms_threshold=float(config.get("yunet_nms_threshold") or 0.3),
            yunet_target=str(config.get("yunet_target") or "cpu"),
            target_crop=None,
        )
        crops = [crop]
    else:
        crops = []
        for index, given in enumerate(target_crops):
            if len(target_crops) == 1:
                crops.append(
                    build_target_crop(
                        target,
                        info,
                        work,
                        model_path=root_path(config, "yunet_model", "models/face_detection_yunet_2023mar.onnx"),
                        sample_stride=int(config.get("sample_stride") or 10),
                        similar_area_ratio=float(config.get("similar_area_ratio") or 0.75),
                        face_index=face_index,
                        score_threshold=float(config.get("yunet_score_threshold") or 0.6),
                        nms_threshold=float(config.get("yunet_nms_threshold") or 0.3),
                        yunet_target=str(config.get("yunet_target") or "cpu"),
                        target_crop=given,
                    )
                )
            else:
                dest = work / f"target_crop_{index}.jpg"
                dest.write_bytes(Path(given).read_bytes())
                crops.append(dest)

    chunks = _split_chunks(target, work, float(config.get("chunk_seconds") or 0))
    swapped_parts = []
    for index, chunk in enumerate(chunks):
        part_out = work / f"ghost_{index:03d}.mp4"
        cmd = video_command(python, sources, crops, chunk.resolve(), part_out.resolve())
        run_ghost(cmd, ghost, _timeout(config))
        if not part_out.is_file():
            raise SwapError(f"GHOST did not write {part_out}")
        swapped_parts.append(part_out)
    swapped = _concat(swapped_parts, work / "ghost.mp4")

    if enhance:
        swapped = enhance_video(
            swapped,
            work,
            gfpgan_python=config.get("gfpgan_python") or None,
            gfpgan_root=config.get("gfpgan_root") or None,
            gfpgan_model=str(root_path(config, "gfpgan_model", "models/GFPGANv1.3.pth")),
            yunet_model=root_path(config, "yunet_model", "models/face_detection_yunet_2023mar.onnx"),
            face_index=face_index,
            score_threshold=float(config.get("yunet_score_threshold") or 0.6),
            nms_threshold=float(config.get("yunet_nms_threshold") or 0.3),
            yunet_target=str(config.get("yunet_target") or "cpu"),
            width=int(info["width"]),
            height=int(info["height"]),
            fps=float(info["fps"]),
        )

    out_info = mux(
        swapped,
        target,
        output,
        crf=crf,
        fps=float(info["fps"]),
        fps_rational=str(info["fps_rational"]),
        duration=float(info["duration"]),
        copy_audio=bool(info["audio_codec"]),
    )
    models = [
        {
            "name": "GHOST",
            "url": "https://github.com/ai-forever/ghost",
            "commit": GHOST_COMMIT,
            "license": "Apache-2.0 code; InsightFace weights are non-commercial research only",
            "entrypoint": "inference.py",
        },
        {
            "name": "YuNet",
            "url": "https://github.com/opencv/opencv_zoo",
            "commit": YUNET_COMMIT,
            "file": "face_detection_yunet_2023mar.onnx",
            "license": "MIT",
        },
    ]
    if enhance:
        models.append(
            {
                "name": "GFPGANv1Clean",
                "url": "https://github.com/TencentARC/GFPGAN",
                "commit": GFPGAN_COMMIT,
                "weights": "GFPGANv1.3.pth",
                "license": "Apache-2.0 except third-party components (StyleGAN2 NVIDIA non-commercial, DFDNet CC-BY-NC-SA-4.0)",
            }
        )
    payload = {
        "consent": True,
        "models": models,
        "licenses": [
            "meme-swap Apache-2.0",
            "GHOST Apache-2.0",
            "YuNet MIT",
            "InsightFace model zoo: non-commercial research purposes only",
        ],
        "config": {**_public_config(config), "crf": crf, "enhance": enhance, "face_index": face_index},
        "ffmpeg": version_line(),
        "ffprobe_in": info["probe"],
        "ffprobe_out": out_info["probe"],
    }
    _sidecar(output, payload)
    return out_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memeswap", description="Local consented face swap for short meme clips")
    parser.add_argument("--version", action="version", version=f"memeswap {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="YAML config. Default: configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("swap-image", help="Swap a source face onto one still")
    image.add_argument("--source", action="append", required=True, type=Path)
    image.add_argument("--target", required=True, type=Path)
    image.add_argument("--out", required=True, type=Path)

    video = sub.add_parser("swap-video", help="Swap a source face onto a clip and copy the original audio")
    video.add_argument("--source", action="append", required=True, type=Path)
    video.add_argument("--target", required=True, type=Path)
    video.add_argument("--out", required=True, type=Path)
    video.add_argument("--crf", type=int, default=None)
    video.add_argument("--consent", action="store_true", help="Required. Only faces you have rights to use.")
    video.add_argument("--face-index", type=int, default=None, help="Which YuNet face to keep when several are similar. 0 is the largest.")
    video.add_argument("--target-crop", action="append", type=Path, default=None)
    video.add_argument("--enhance", action="store_true", help="Run GFPGANv1Clean. Off by default. Does not use RetinaFace.")

    bench = sub.add_parser("benchmark", help="Time a swap of the first N seconds and print the GPU name")
    bench.add_argument("--seconds", type=float, required=True)
    bench.add_argument("--source", action="append", required=True, type=Path)
    bench.add_argument("--target", required=True, type=Path)
    bench.add_argument("--consent", action="store_true", help="Required. Only faces you have rights to use.")
    bench.add_argument("--crf", type=int, default=None)
    bench.add_argument("--face-index", type=int, default=None)
    bench.add_argument("--target-crop", action="append", type=Path, default=None)
    bench.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "swap-image":
            swap_image(args.source, args.target, args.out, config)
            print(args.out)
            return 0
        if args.command == "swap-video":
            if not args.consent:
                parser.error("swap-video requires --consent. Only swap faces you have rights to use.")
            crf = args.crf if args.crf is not None else int(config.get("crf") or 18)
            swap_video(
                args.source,
                args.target,
                args.out,
                config,
                crf=crf,
                face_index=args.face_index,
                target_crops=args.target_crop or [],
                enhance=bool(args.enhance),
                consent=True,
            )
            print(args.out)
            return 0
        if args.command == "benchmark":
            if not args.consent:
                parser.error("benchmark requires --consent. Only swap faces you have rights to use.")
            info = probe(args.target)
            root = Path(config["_root"])
            work = root_path(config, "work_dir", "work") / _job_id(args.target)
            work.mkdir(parents=True, exist_ok=True)
            trimmed = work / f"benchmark_{args.seconds:g}s.mp4"
            trim_cmd = [
                tool("ffmpeg"),
                "-y",
                "-i",
                str(args.target),
                "-t",
                str(args.seconds),
                "-c:v",
                "libx264",
            ]
            if info["audio_codec"]:
                trim_cmd += ["-c:a", "copy"]
            else:
                trim_cmd += ["-an"]
            trim_cmd.append(str(trimmed))
            trim = run(trim_cmd)
            if trim.returncode != 0:
                raise MuxError(trim.stderr.strip() or "failed to trim the benchmark clip")
            out = args.out or (work / "benchmark.mp4")
            crf = args.crf if args.crf is not None else int(config.get("crf") or 18)
            started = time.perf_counter()
            swap_video(
                args.source,
                trimmed,
                out,
                config,
                crf=crf,
                face_index=args.face_index,
                target_crops=args.target_crop or [],
                enhance=False,
                consent=True,
            )
            elapsed = time.perf_counter() - started
            print(f"elapsed_seconds {elapsed:.3f}")
            print(f"input_seconds {info['duration']:.3f}")
            print(f"benchmark_seconds {args.seconds:g}")
            print(_gpu_report())
            print(out)
            return 0
    except (IngestError, DetectError, SwapError, MuxError, EnhanceError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
