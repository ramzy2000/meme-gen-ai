from __future__ import annotations

import pytest

from memeswap.cli import main
from memeswap.swap import image_command, video_command


def test_swap_video_help() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["swap-video", "--help"])
    assert caught.value.code == 0


def test_swap_video_requires_consent(tmp_path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "swap-video",
                "--source",
                str(tmp_path / "face.jpg"),
                "--target",
                str(tmp_path / "clip.mp4"),
                "--out",
                str(tmp_path / "out.mp4"),
            ]
        )
    assert caught.value.code == 2


def test_commands_use_existing_ghost_flags(tmp_path) -> None:
    video = video_command(
        "python",
        [tmp_path / "face.jpg"],
        [tmp_path / "crop.jpg"],
        tmp_path / "clip.mp4",
        tmp_path / "ghost.mp4",
    )
    assert "--target_video" in video
    assert "--use_sr" not in video
    assert "--image_to_image" not in video
    image = image_command("python", [tmp_path / "face.jpg"], tmp_path / "frame.png", tmp_path / "out.png")
    assert "--image_to_image" in image
    assert image[image.index("--image_to_image") + 1] == "True"
    assert "--target_video" not in image
