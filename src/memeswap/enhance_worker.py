"""Run GFPGANv1Clean on already-cropped faces.

This process must not import facexlib, GFPGANer, or inference_gfpgan.py.
GFPGANer hardcodes det_model='retinaface_resnet50'.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


def _load_clean_arch(gfpgan_root: Path):
    pkg = Path(gfpgan_root) / "gfpgan"
    arch = pkg / "archs" / "gfpganv1_clean_arch.py"
    if not arch.is_file():
        raise SystemExit(
            f"GFPGAN checkout at {gfpgan_root} has no gfpgan/archs/gfpganv1_clean_arch.py. "
            "Clone https://github.com/TencentARC/GFPGAN at 7552a7791caad982045a7bbe5634bbf1cd5c8679."
        )
    # Pre-seed packages so gfpgan/__init__.py does not import utils.py (and facexlib).
    gfpgan_mod = types.ModuleType("gfpgan")
    gfpgan_mod.__path__ = [str(pkg)]
    gfpgan_mod.__package__ = "gfpgan"
    sys.modules["gfpgan"] = gfpgan_mod
    archs_mod = types.ModuleType("gfpgan.archs")
    archs_mod.__path__ = [str(pkg / "archs")]
    archs_mod.__package__ = "gfpgan.archs"
    sys.modules["gfpgan.archs"] = archs_mod

    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    banned = [
        name
        for name in sys.modules
        if name == "facexlib" or name.startswith("facexlib.") or "retinaface" in name.lower()
    ]
    if banned:
        raise SystemExit("refusing to run because facexlib or RetinaFace was imported: " + ", ".join(banned))
    return GFPGANv1Clean


def _restore_one(net, image_path: Path, out_path: Path, device: str) -> None:
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.BICUBIC)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)
    with torch.no_grad():
        output = net(tensor, return_rgb=False)[0]
    output = output.squeeze(0).detach().float().cpu().clamp(-1, 1)
    output = ((output + 1.0) / 2.0 * 255.0).permute(1, 2, 0).numpy().astype("uint8")
    Image.fromarray(output, mode="RGB").save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore 512px face crops with GFPGANv1Clean")
    parser.add_argument("--gfpgan-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch

    GFPGANv1Clean = _load_clean_arch(Path(args.gfpgan_root))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    loadnet = torch.load(args.model, map_location=device)
    key = "params_ema" if isinstance(loadnet, dict) and "params_ema" in loadnet else "params"
    if not isinstance(loadnet, dict) or key not in loadnet:
        raise SystemExit(f"{args.model} does not contain params_ema or params")
    net.load_state_dict(loadnet[key], strict=True)
    net.eval()
    net.to(device)

    crops = sorted(Path(args.crops).glob("*.png"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for crop in crops:
        _restore_one(net, crop, out_dir / crop.name, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
