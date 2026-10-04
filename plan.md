# Expert prompt: local GPU meme video face swap

Paste everything below the line into Grok Expert.

---

You are a staff ML engineer designing a local, GPU-only meme-video pipeline. Research current open-source systems before recommending anything. Do not invent model names, licenses, or VRAM numbers. Cite the repo and the license for every dependency you recommend.

## Product

We generate short meme clips by editing existing video, not by text-to-video from scratch. V1 is identity replacement: take a source face (one or more stills) and a target clip, replace one or more people, keep the original motion, lighting, background, and audio, and write a shareable MP4. Later phases may add lip sync to new audio, expression transfer, and simple caption/overlay templates. We are not training a foundation model.

## Constraints

- 100% local inference. No cloud APIs, no uploads.
- NVIDIA CUDA first. State a CPU fallback only as a degraded path.
- Tier the design for 8 GB, 12 GB, and 24 GB VRAM. Call out what is realistic on each.
- We will own a git repo. Prefer a thin orchestration layer over a maintained project rather than a from-scratch model, unless research shows that is the wrong call.
- Python for the ML path. A small CLI is enough for v1; a web UI is optional and must not block the CLI.
- Commercial use is possible later. Flag non-commercial weights (InsightFace inswapper and similar) and prefer stacks we can actually ship.

## Research, then decide

Compare at least: FaceFusion (current 3.x, including headless/job mode), InsightFace / inswapper and newer 256 swapper weights, Roop-Unleashed, ReActor / ComfyUI-ReActor, Deep-Live-Cam, and DeepFaceLab. Also check whether identity-preserving video models (Wan-Animate, Stand-In, LivePortrait) beat classic swap for meme clips, and at what VRAM cost. For each: maintenance status, license, one-shot vs per-pair training, video + audio handling, multi-face, occlusion, and realistic minutes-per-minute on a consumer GPU.

Recommend one v1 stack. Default bias: wrap FaceFusion or an InsightFace ONNX pipeline, restore with GFPGAN or CodeFormer, mux audio with ffmpeg. Reject DeepFaceLab for v1 unless we need a fixed celebrity pair and can afford training. Reject archived Roop.

## Pipeline to specify

1. Ingest: mp4/mov/webm, probe with ffprobe, reject odd codecs.
2. Scene/face track: detect, align, track identities across frames so a swap does not flicker between people.
3. Swap: source embedding to target track. Expose blend strength.
4. Restore and blend: face enhancer, color match, temporal smoothing so the face does not shimmer.
5. Encode: original audio copied, H.264 or HEVC, configurable crf. Chunk long clips so RAM stays bounded.
6. Cache: face embeddings and per-video tracks on disk.

## Repo shape

Propose a layout (`src/`, `configs/`, `models/` gitignored, `scripts/`, `tests/`, Dockerfile or conda env, README with exact install). Pin Python, CUDA, and ffmpeg. One command for image swap, one for video swap, one benchmark. Document model download, license files, and a consent note: only faces the operator has rights to use.

## Quality bar

Define failure cases: profile faces, glasses, hands over mouth, fast motion, tiny faces, multiple similar people. Specify the metrics we should eyeball (identity similarity, flicker, edge halo) and the settings to tune first.

## Deliverables

1. Recommendation in one paragraph, with the rejected alternatives and why.
2. Architecture diagram in text.
3. Exact stack, licenses, and VRAM table.
4. Repo skeleton and the three CLI commands.
5. A phased plan: v1 face-swap meme clip, v2 lip sync, v3 batch templates. Estimate effort for a two-person team that already ships Python and .NET.
6. Top five setup footguns (CUDA/onnxruntime mismatch, weight licenses, VRAM with the enhancer on, audio drift, temporal flicker).

Do not write a tutorial for non-consensual or deceptive use. Assume meme edits of consented or public-domain material.