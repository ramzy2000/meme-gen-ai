# memeswap

Local identity replacement for short meme clips. One or more source stills, one target clip, write an MP4 that keeps the original motion, lighting, background, and audio.

Use this only on faces you have rights to use. `swap-video` and `benchmark` refuse to run unless you pass `--consent`. There is no cloud upload and no workflow for passing off a swap as the untouched original.

## What fails

Profile faces, glasses, a hand over the mouth, fast motion, tiny faces, and two people of similar size. Two similar faces abort unless you pass `--face-index` (0 is the largest box). Look at identity stability, flicker, and edge halo. The first knobs are `--face-index`, `--target-crop`, and leaving the enhancer off.

`configs/default.yaml` is the CUDA config and the enhancer is off. `configs/cpu.yaml` selects the ONNX Runtime CPU install for GHOST's detector. It does not make the generator run on CPU: `inference.py` calls `G.cuda()` and `G.half()`. This repo does not publish a VRAM table. Time a real clip with `memeswap benchmark` and record the card it prints.

## Install

GHOST pins `torch==1.6.0+cu101`. Wheels exist for CPython 3.6, 3.7, and 3.8. The 3.9, 3.10, 3.11, and 3.12 wheel URLs are not published (HTTP 403). Use Python 3.8 for the environment that installs GHOST. Do not upgrade that torch pin. The CLI itself imports on newer CPython; the swap subprocess must be the 3.8 interpreter (`ghost_python` in the config).

The torch pin is `1.6.0+cu101` (CUDA 10.1). `onnxruntime-gpu==1.4.0` is not on PyPI. If that install fails, install `onnxruntime` instead. That is the degraded CPU ONNX path. The generator still needs CUDA.

```bash
git clone <this repo>
cd meme-gen-ai
git submodule update --init --recursive
git apply --directory=third_party/ghost patches/ghost-skip-audio-mux.patch
conda env create -f environment.yml
conda activate memeswap
pip install -e .
pip install -r third_party/ghost/requirements.txt
```

If `onnxruntime-gpu==1.4.0` fails:

```bash
pip install onnxruntime
```

`download_models.sh` is `sh` plus `wget`. On Windows, Git Bash often has no `wget`. `scripts/download_models.py` runs the script when `wget` exists, and otherwise downloads the same URLs the script lists.

```bash
python scripts/download_models.py
```

ffmpeg and ffprobe must be on PATH. Prefer an LGPL build. The final video step uses `libx264`, and builds that include libx264 are commonly configured with `--enable-gpl`. Audio is `-c:a copy` and is not re-encoded. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

GHOST's own audio helper is not used. The one-line patch in `patches/ghost-skip-audio-mux.patch` comments out `add_audio_from_another_video`, which re-encodes audio and calls `rm` / `mv`. If `git apply` reports that the patch is already present, leave the file as it is.

`pip install` pulls `opencv-python` 4.x. `face_detection_yunet_2023mar.onnx` is the fixed-shape model for that DNN backend. The 2026may YuNet file is for OpenCV 5 and is not used.

## Commands

```bash
memeswap swap-image --source face.jpg --target frame.png --out out.png
memeswap swap-video --source face.jpg --target clip.mp4 --out out.mp4 --crf 18 --consent
memeswap benchmark --seconds 10 --source face.jpg --target clip.mp4 --consent
```

`benchmark` also needs `--source`, `--target`, and `--consent`. It trims the first N seconds, runs the same swap, and prints wall time plus the GPU name from `nvidia-smi` when that command exists. No sample face clip is shipped.

Accepted containers are mp4, mov, and webm. Clips with no video stream, a codec ffmpeg cannot decode, or a variable frame rate (`r_frame_rate` different from `avg_frame_rate`) are rejected. A JSON sidecar is written next to the MP4 with the models, licenses, config, `consent: true`, and ffprobe for the input and the output.

More than one `--source` needs the same number of `--target-crop` stills, in order. There is no blend flag. GHOST's `inference.py` does not have one, and `--use_sr` is not passed.

## Enhancer

Off unless you pass `--enhance`. That path loads `GFPGANv1Clean` and `GFPGANv1.3.pth` only. It does not call `inference_gfpgan.py` or `GFPGANer`, because those construct a RetinaFace detector (`det_model='retinaface_resnet50'`).

GFPGAN's license is Apache-2.0 except third-party components, including StyleGAN2 under the NVIDIA non-commercial terms and DFDNet under CC-BY-NC-SA 4.0. GFPGAN wants `torch>=1.7`, which does not match GHOST's torch pin, so set `gfpgan_python` and `gfpgan_root` to a separate checkout of TencentARC/GFPGAN at `7552a7791caad982045a7bbe5634bbf1cd5c8679`. Weights are downloaded only with:

```bash
python scripts/download_models.py --with-gfpgan
```

## Weights that are not Apache-2.0

GHOST's code license is Apache-2.0. The default swap still loads InsightFace weights (`scrfd_10g_bnkps.onnx`, `glintr100.onnx`, ArcFace `backbone.pth`, and the vendored `2d106det` checkpoint). The InsightFace model zoo README says: "ALL models are available for non-commercial research purposes only." Our tracker does not use those files; YuNet does. Shipping a working swap is not a permissive redistribution of those weights.

`third_party/ghost/coordinate_reg/model/SCRFD/model_25GF.pth` is in the GHOST tree. `inference.py` does not reference it. AdaptiveWingLoss, the GHOST submodule, has no license file at its pin and is not called.

GHOST 2.0 is not used. Its submodules include BlazeFace, stylematte, and DECA.

## Tests

```bash
pip install -e ".[test]"
pytest
memeswap swap-video --help
```

The tests build short fixtures with ffmpeg. They do not download weights and they do not need a GPU.
