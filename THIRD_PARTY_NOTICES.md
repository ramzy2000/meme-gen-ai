# Third-party notices

Quotes below are from the license or notice files at the pinned commits. Where a license file and a README disagree, the license file is the one recorded. Weights that have no license file of their own are recorded with the notice that ships next to them.

This project's own code is Apache-2.0. See [LICENSE](LICENSE).

## GHOST

- URL: https://github.com/ai-forever/ghost
- Pin: `44e58aad8600ee83ad4c8213aaa9698ccaf66c1c`
- License file: `LICENSE` at that commit, Apache License Version 2.0
- Note: Video swap calls `python inference.py --source_paths ... --target_faces_paths ... --target_video ...`. One local patch skips `add_audio_from_another_video` so audio is copied by our ffmpeg step. See [patches/ghost-skip-audio-mux.patch](patches/ghost-skip-audio-mux.patch).

`requirements.txt` at that pin does not select a Python version. It pins `torch==1.6.0+cu101`, `torchvision==0.7.0+cu101`, `onnxruntime-gpu==1.4.0`, `insightface==0.2.1`, and `mxnet-cu101mkl`. Wheels for `torch-1.6.0+cu101` exist for `cp36`, `cp37`, and `cp38` (HTTP 200). The `cp39`, `cp310`, and `cp311` URLs return HTTP 403. PyPI has no `onnxruntime-gpu==1.4.0` release. The GHOST README says: if `onnxruntime-gpu` cannot be installed, try `onnxruntime`. That CPU ONNX build is a degraded path. `inference.py` still calls `G.cuda()` and `G.half()`.

## AdaptiveWingLoss

- URL: https://github.com/NastyaMittseva/AdaptiveWingLoss
- Pin: `6a6e42fcc435dc8fd604f211390099725a4222a6` (GHOST submodule)
- License file: none at that commit
- Note: Not imported by `inference.py`. Not on the default path. `download_models.sh` can place a training checkpoint under `AdaptiveWingLoss/AWL_detector`; do not treat that as part of swap.

## apex (vendored inside GHOST)

- Path: `third_party/ghost/apex/LICENSE` at the GHOST pin
- License: the file is the 3-clause BSD text. It begins `All rights reserved.`
- Note: `inference.py` does not import apex. Not on the default path.

## InsightFace weights used by the GHOST entrypoint

- Code package pin from GHOST `requirements.txt`: `insightface==0.2.1`
- Notice: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- Quote: `ALL models are available for non-commercial research purposes only.`
- `python-package/LICENSE` returned 404, and the InsightFace repository root listing has no `LICENSE` file. GHOST's Apache-2.0 `LICENSE` does not mention these weights.

`download_models.sh` fetches these files into the GHOST tree (not into `models/`):

- `insightface_func/models/antelope/scrfd_10g_bnkps.onnx`
- `insightface_func/models/antelope/glintr100.onnx`
- `arcface_model/backbone.pth`
- `arcface_model/iresnet.py`

`inference.py` always constructs `Face_detect_crop(name='antelope')` and loads `arcface_model/backbone.pth`. The GHOST tree also vendors `coordinate_reg/model/2d106det`, which `Handler` loads while pasting the swap. Those files are on the default swap path. A working swap is not a permissive-weight redistribution.

`coordinate_reg/model/SCRFD/model_25GF.pth` is in the GHOST tree. No Python file at this pin references that filename, and our tracker does not load it.

Our detector is YuNet. This repo does not vendor inswapper, buffalo_l, RetinaFace, or a separate SCRFD package.

## YuNet

- URL: https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet
- Pin: `47534e27c9851bb1128ccc0102f1145e27f23f98`
- License file: `models/face_detection_yunet/LICENSE`, MIT
- Quote: `Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>`
- Note: The file we download is `face_detection_yunet_2023mar.onnx`, used with `cv2.FaceDetectorYN.create`. The zoo README says `face_detection_yunet_2026may.onnx` is the default for the OpenCV 5 ONNX Runtime engine. We do not use that file on the OpenCV 4 DNN path.

## GFPGAN

- URL: https://github.com/TencentARC/GFPGAN
- Pin: `7552a7791caad982045a7bbe5634bbf1cd5c8679`
- License file: `LICENSE`
- Quote: `GFPGAN is licensed under the Apache License Version 2.0 except for the third-party components listed below.`
- The same file lists StyleGAN2 under the NVIDIA license. Quote from section 3.3: `The Work and any derivative works thereof only may be used or intended for use non-commercially.` The same section says non-commercially means research or evaluation purposes only.
- The same file says: `The codes are largely modified from the repository DFDNet. Their license is Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.`
- `basicsr` is listed in that file as Apache-2.0.
- Note: Enhancer is off in `configs/default.yaml` and `configs/cpu.yaml`. `--enhance` loads `GFPGANv1Clean` and `GFPGANv1.3.pth` (`https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth`) only. It does not call `inference_gfpgan.py` or `GFPGANer`. `gfpgan/utils.py` always builds `FaceRestoreHelper` with `det_model='retinaface_resnet50'`. GFPGAN's `requirements.txt` requires `torch>=1.7`, which conflicts with GHOST's `torch==1.6.0+cu101`, so the enhancer uses a separate interpreter and is not installed by default.

## GHOST 2.0

Not included. https://github.com/ai-forever/ghost-2.0 `.gitmodules` pulls `hollance/BlazeFace-PyTorch`, `chroneus/stylematte`, and `yfeng95/DECA`.

## ffmpeg

Not vendored. The binary on PATH is an external tool. Prefer an LGPL build. H.264 encoding with `libx264` is commonly a GPL build (`--enable-gpl`).

The build used while writing the tests printed:

`ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers`

and its configuration line includes `--enable-gpl` and `--enable-libx264`. That is a GPL build, not an LGPL build. Audio is still `-c:a copy` and is not re-encoded.
