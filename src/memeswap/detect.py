"""YuNet face sampling for the target crop and the ambiguity check.

GHOST's own inference still loads its antelope detector. This module is only
the tracker that chooses which person we pass in as --target_faces_paths.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from memeswap.ingest import frame_count


class DetectError(RuntimeError):
    pass


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def frame_is_ambiguous(faces: list[Face], similar_area_ratio: float) -> bool:
    """True when the second-largest face is within ``similar_area_ratio`` of the largest."""
    if len(faces) < 2:
        return False
    largest = faces[0].area
    if largest <= 0:
        return False
    return (faces[1].area / largest) >= similar_area_ratio


def sample_indices(n_frames: int, stride: int) -> list[int]:
    if n_frames <= 0:
        return []
    step = max(int(stride), 1)
    indices = list(range(0, n_frames, step))
    return indices or [0]


def _load_detector(model_path: Path, width: int, height: int, score: float, nms: float, target: str):
    import cv2

    model_path = Path(model_path)
    if not model_path.is_file():
        raise DetectError(
            f"YuNet model not found at {model_path}. Run scripts/download_models.py."
        )
    head = model_path.read_bytes()[:80]
    if head.startswith(b"version https://git-lfs"):
        raise DetectError(
            f"{model_path} is a Git LFS pointer, not an ONNX file. "
            "Run scripts/download_models.py again."
        )
    if target == "cuda":
        backend_id = cv2.dnn.DNN_BACKEND_CUDA
        target_id = cv2.dnn.DNN_TARGET_CUDA
    else:
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
    return cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(width, height),
        score_threshold=float(score),
        nms_threshold=float(nms),
        top_k=5000,
        backend_id=backend_id,
        target_id=target_id,
    )


def _faces_in_frame(detector, frame) -> list[Face]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, found = detector.detect(frame)
    if found is None:
        return []
    faces = []
    for row in found:
        faces.append(
            Face(
                x=float(row[0]),
                y=float(row[1]),
                w=float(row[2]),
                h=float(row[3]),
                score=float(row[-1]),
            )
        )
    faces.sort(key=lambda face: face.area, reverse=True)
    return faces


def _crop(frame, face: Face, margin: float = 0.5):
    height, width = frame.shape[:2]
    cx = face.x + face.w / 2.0
    cy = face.y + face.h / 2.0
    side = max(face.w, face.h) * (1.0 + margin)
    x0 = max(0, int(round(cx - side / 2.0)))
    y0 = max(0, int(round(cy - side / 2.0)))
    x1 = min(width, int(round(cx + side / 2.0)))
    y1 = min(height, int(round(cy + side / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise DetectError("the selected face crop fell outside the frame")
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def build_target_crop(
    video: Path,
    info: dict,
    work_dir: Path,
    *,
    model_path: Path,
    sample_stride: int,
    similar_area_ratio: float,
    face_index: int | None,
    score_threshold: float,
    nms_threshold: float,
    yunet_target: str,
    target_crop: Path | None = None,
) -> Path:
    """Write ``work_dir/target_crop.jpg`` and ``work_dir/tracks.json``."""
    import cv2

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_crop = work_dir / "target_crop.jpg"

    if target_crop is not None:
        source = Path(target_crop)
        if not source.is_file():
            raise DetectError(f"--target-crop does not exist: {source}")
        image = cv2.imread(str(source))
        if image is None:
            raise DetectError(f"could not read --target-crop: {source}")
        cv2.imwrite(str(out_crop), image)
        (work_dir / "tracks.json").write_text(
            json.dumps(
                {
                    "video": str(Path(video).resolve()),
                    "override": str(source.resolve()),
                    "face_index": face_index,
                    "frames": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_crop

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise DetectError(f"OpenCV could not open {video}")
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        n_frames = frame_count(info)
    indices = sample_indices(n_frames, sample_stride)
    detector = _load_detector(
        model_path,
        int(info["width"]),
        int(info["height"]),
        score_threshold,
        nms_threshold,
        yunet_target,
    )

    sampled = []
    ambiguous = False
    chosen_index = 0 if face_index is None else face_index
    best_frame = None
    best_face: Face | None = None
    best_area = -1.0
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            faces = _faces_in_frame(detector, frame)
            sampled.append(
                {
                    "index": index,
                    "faces": [asdict(face) for face in faces],
                }
            )
            if frame_is_ambiguous(faces, similar_area_ratio):
                ambiguous = True
            if chosen_index < len(faces) and faces[chosen_index].area > best_area:
                best_area = faces[chosen_index].area
                best_frame = frame
                best_face = faces[chosen_index]
    finally:
        capture.release()

    tracks = {
        "video": str(Path(video).resolve()),
        "sample_stride": sample_stride,
        "similar_area_ratio": similar_area_ratio,
        "face_index": face_index,
        "ambiguous": ambiguous,
        "frames": sampled,
    }
    (work_dir / "tracks.json").write_text(json.dumps(tracks, indent=2), encoding="utf-8")

    if ambiguous and face_index is None:
        raise DetectError(
            "two faces of similar size are in the clip. "
            "Refusing to guess; pass --face-index (0 is the largest face)."
        )
    if best_frame is None or best_face is None:
        raise DetectError(
            "YuNet did not find the selected face. "
            "Pass --target-crop with a still of the person to replace."
        )
    cv2.imwrite(str(out_crop), _crop(best_frame, best_face)[0])
    return out_crop
