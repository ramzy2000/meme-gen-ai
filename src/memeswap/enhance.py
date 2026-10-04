"""Optional GFPGANv1Clean pass. Off unless --enhance is set.

Does not call inference_gfpgan.py or GFPGANer, because that helper always
constructs FaceRestoreHelper with det_model='retinaface_resnet50'.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from memeswap.detect import _crop, _faces_in_frame, _load_detector


class EnhanceError(RuntimeError):
    pass


def enhance_video(
    swapped_video: Path,
    work_dir: Path,
    *,
    gfpgan_python: str | None,
    gfpgan_root: str | None,
    gfpgan_model: str | None,
    yunet_model: Path,
    face_index: int | None,
    score_threshold: float,
    nms_threshold: float,
    yunet_target: str,
    width: int,
    height: int,
    fps: float,
) -> Path:
    if not gfpgan_python:
        raise EnhanceError(
            "gfpgan_python is unset, so the enhancer stays off and no GFPGAN weights were downloaded. "
            "GFPGAN requires torch>=1.7, which conflicts with GHOST's pin torch==1.6.0+cu101. "
            "Point gfpgan_python at a separate interpreter before passing --enhance."
        )
    if not gfpgan_root or not Path(gfpgan_root).is_dir():
        raise EnhanceError(
            "gfpgan_root is unset or missing. Clone https://github.com/TencentARC/GFPGAN "
            "at 7552a7791caad982045a7bbe5634bbf1cd5c8679 and set gfpgan_root. "
            "Do not run inference_gfpgan.py."
        )
    model = Path(gfpgan_model) if gfpgan_model else None
    if model is None or not model.is_file():
        raise EnhanceError(
            "GFPGANv1.3.pth is missing. Run: python scripts/download_models.py --with-gfpgan"
        )

    import cv2

    work = Path(work_dir) / "enhance"
    crops = work / "crops"
    restored = work / "restored"
    crops.mkdir(parents=True, exist_ok=True)
    restored.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(swapped_video))
    if not capture.isOpened():
        raise EnhanceError(f"could not read swapped video {swapped_video}")
    detector = _load_detector(
        Path(yunet_model), width, height, score_threshold, nms_threshold, yunet_target
    )
    chosen = 0 if face_index is None else face_index
    boxes: list[tuple[int, int, int, int] | None] = []
    index = 0
    frame_size = None
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
        faces = _faces_in_frame(detector, frame)
        if chosen < len(faces):
            crop, box = _crop(frame, faces[chosen])
            cv2.imwrite(str(crops / f"{index:06d}.png"), crop)
            boxes.append(box)
        else:
            boxes.append(None)
        index += 1
    capture.release()
    if index == 0 or frame_size is None:
        raise EnhanceError(f"{swapped_video} has no frames to restore")

    worker = Path(__file__).resolve().parent / "enhance_worker.py"
    proc = subprocess.run(
        [
            gfpgan_python,
            str(worker),
            "--gfpgan-root",
            str(gfpgan_root),
            "--model",
            str(model),
            "--crops",
            str(crops),
            "--out",
            str(restored),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise EnhanceError(f"GFPGANv1Clean worker exited {proc.returncode}\n{detail}")

    out_path = work / "enhanced.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size,
    )
    if not writer.isOpened():
        raise EnhanceError(f"could not write {out_path}")
    capture = cv2.VideoCapture(str(swapped_video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        box = boxes[index] if index < len(boxes) else None
        restored_path = restored / f"{index:06d}.png"
        if box is not None and restored_path.is_file():
            face_img = cv2.imread(str(restored_path))
            if face_img is not None:
                x0, y0, x1, y1 = box
                x0 = max(0, min(frame.shape[1] - 1, x0))
                y0 = max(0, min(frame.shape[0] - 1, y0))
                x1 = min(frame.shape[1], x1)
                y1 = min(frame.shape[0], y1)
                if x1 > x0 and y1 > y0:
                    frame = frame.copy()
                    frame[y0:y1, x0:x1] = cv2.resize(face_img, (x1 - x0, y1 - y0))
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    return out_path
