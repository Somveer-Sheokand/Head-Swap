"""
Keep the template's ORIGINAL hair pixel-identical after a face swap: only the face comes from the generation.

The hair is found automatically (nothing hardcoded): inside the region the swap changed, it is the dark area
connected to the top of the head in the ORIGINAL template. Thin dark lines (glasses frames) are removed with
an opening so they don't count as hair. Those original hair pixels are pasted back with a soft edge.

  python keep_hair.py TEMPLATE SWAPPED [SWAPPED ...] --out-dir DIR
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def hair_mask(tpl: Image.Image, swapped: Image.Image, dark: int = 70) -> np.ndarray:
    t = np.asarray(tpl.convert("RGB"), np.int16)
    s = np.asarray(swapped.convert("RGB"), np.int16)
    changed = np.abs(t - s).max(2) > 6
    changed = ndimage.binary_dilation(ndimage.binary_closing(changed, iterations=4), iterations=25)
    ys, xs = np.nonzero(changed)
    if len(ys) == 0:
        return np.zeros(changed.shape, bool)
    region = np.zeros_like(changed)
    region[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = True

    lum = t.mean(2)
    dark_px = (lum < dark) & region
    dark_px = ndimage.binary_opening(dark_px, iterations=3)  # drop glasses frames / thin lines
    labels, n = ndimage.label(dark_px)
    if n == 0:
        return np.zeros(changed.shape, bool)
    # hair = dark components that reach the top part of the changed head region
    top = ys.min() + int(0.25 * (ys.max() - ys.min()))
    keep = [i for i in range(1, n + 1) if np.nonzero(labels == i)[0].min() <= top]
    hair = np.isin(labels, keep)
    return ndimage.binary_fill_holes(ndimage.binary_closing(hair, iterations=3))


def face_mask(tpl: Image.Image, swapped: Image.Image) -> np.ndarray:
    """The template's face + neck skin (where the new face goes). Hair, ears' surroundings, background and
    clothes are NOT in it, so they stay original."""
    t = np.asarray(tpl.convert("RGB"), np.int16)
    s = np.asarray(swapped.convert("RGB"), np.int16)
    changed = np.abs(t - s).max(2) > 6
    changed = ndimage.binary_closing(changed, iterations=4)
    ys, xs = np.nonzero(changed)
    if len(ys) == 0:
        return np.zeros(changed.shape, bool)
    region = np.zeros_like(changed)
    pad = 20
    region[max(0, ys.min() - pad): ys.max() + pad, max(0, xs.min() - pad): xs.max() + pad] = True

    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    skin = (r > 90) & (r > g) & (g >= b * 0.85) & (r - b > 25) & region
    skin = ndimage.binary_opening(skin, iterations=2)
    labels, n = ndimage.label(skin)
    if n == 0:
        return np.zeros(changed.shape, bool)
    sizes = ndimage.sum(skin, labels, range(1, n + 1))
    face = labels == (int(np.argmax(sizes)) + 1)  # biggest skin blob = face + neck
    # include the eyes, brows, glasses and lips inside the face (holes in the skin mask)
    face = ndimage.binary_fill_holes(ndimage.binary_closing(face, iterations=8))
    face = face & ~hair_mask(tpl, swapped)  # never take hair from the swap
    return ndimage.binary_erosion(face, iterations=1)


def keep_hair(tpl_path: str, swapped_path: str, out_path: str):
    """Original template everywhere except the face skin, which comes from the swap."""
    tpl = Image.open(tpl_path).convert("RGB")
    sw = Image.open(swapped_path).convert("RGB").resize(tpl.size)
    m = face_mask(tpl, sw)
    mask = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    out = tpl.copy()
    out.paste(sw, (0, 0), mask)
    out.save(out_path)
    return m.mean()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("template")
    p.add_argument("swapped", nargs="+")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.swapped:
        frac = keep_hair(a.template, f, os.path.join(a.out_dir, os.path.basename(f)))
        print(f"{os.path.basename(f)}: face taken from the swap on {frac * 100:.2f}% of pixels")
