"""Download GHOST weights and the YuNet ONNX.

Runs third_party/ghost/download_models.sh when wget is available.
Otherwise downloads the same wget URLs the script lists. Rejects a Git LFS pointer.
GFPGAN weights are downloaded only with --with-gfpgan.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GHOST = ROOT / "third_party" / "ghost"
YUNET_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        f"{YUNET_COMMIT}/models/face_detection_yunet/{YUNET_NAME}"
    ),
    (
        "https://github.com/opencv/opencv_zoo/raw/"
        f"{YUNET_COMMIT}/models/face_detection_yunet/{YUNET_NAME}"
    ),
]
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
WGET_LINE = re.compile(r"^wget\s+-P\s+(\S+)\s+(\S+)\s*$")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _bash() -> str | None:
    found = shutil.which("bash")
    if found:
        return found
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    return None


def _wget_available(bash: str) -> bool:
    proc = subprocess.run(
        [bash, "-lc", "command -v wget"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "memeswap"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    head = dest.read_bytes()[:80]
    if head.startswith(LFS_PREFIX) or dest.stat().st_size < 1000 and YUNET_NAME in dest.name:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"refusing Git LFS pointer from {url}")


def _wget_targets(script: Path) -> list[tuple[Path, str]]:
    targets = []
    for line in script.read_text(encoding="utf-8").splitlines():
        match = WGET_LINE.match(line.strip())
        if not match:
            continue
        folder = (GHOST / match.group(1)).resolve()
        url = match.group(2)
        name = url.rstrip("/").split("/")[-1]
        targets.append((folder / name, url))
    if not targets:
        raise SystemExit(f"no wget lines found in {script}")
    return targets


def download_ghost() -> None:
    script = GHOST / "download_models.sh"
    if not script.is_file():
        raise SystemExit(
            "third_party/ghost/download_models.sh is missing. "
            "Run: git submodule update --init --recursive"
        )
    bash = _bash()
    if bash and _wget_available(bash):
        proc = subprocess.run([bash, str(script.name)], cwd=str(GHOST), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"download_models.sh exited {proc.returncode}")
        return
    print(
        "wget is not available, so download_models.sh was not run. "
        "Downloading the URLs listed in that script.",
        file=sys.stderr,
    )
    for dest, url in _wget_targets(script):
        if dest.is_file() and dest.stat().st_size > 1000:
            print(f"exists {dest}")
            continue
        print(f"get {url}")
        _download(url, dest)


def download_yunet() -> None:
    dest = ROOT / "models" / YUNET_NAME
    if dest.is_file() and not dest.read_bytes()[:80].startswith(LFS_PREFIX) and dest.stat().st_size > 1000:
        print(f"exists {dest}")
        return
    errors = []
    for url in YUNET_URLS:
        print(f"get {url}")
        try:
            _download(url, dest)
            print(f"wrote {dest}")
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise SystemExit("YuNet download failed\n" + "\n".join(errors))


def download_gfpgan() -> None:
    dest = ROOT / "models" / "GFPGANv1.3.pth"
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"exists {dest}")
        return
    print(f"get {GFPGAN_URL}")
    _download(GFPGAN_URL, dest)
    print(f"wrote {dest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download YuNet and the GHOST weights listed in download_models.sh")
    parser.add_argument(
        "--with-gfpgan",
        action="store_true",
        help="Also download GFPGANv1.3.pth. Off by default.",
    )
    args = parser.parse_args(argv)
    download_yunet()
    download_ghost()
    if args.with_gfpgan:
        download_gfpgan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
