"""Image perturbations shared by demo.py and stain_robustness.py: H&E stain
shift (colour deconvolution, random per-channel scale and offset), Gaussian
blur, and JPEG re-compression."""

import io

import numpy as np
from PIL import Image, ImageFilter

_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def stain_shift(img: Image.Image, strength: float, seed: int = 0) -> Image.Image:
    if strength == 0:
        return img
    rng = np.random.default_rng(seed)
    rgb = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0
    od = -np.log(np.clip(rgb, 1e-6, 1.0))
    hed = od @ _HED_FROM_RGB
    alpha = 1 + rng.uniform(-strength, strength, 3)
    beta = rng.uniform(-strength, strength, 3) * 0.1
    hed = hed * alpha + beta
    out = np.exp(-(hed @ _RGB_FROM_HED))
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img if radius == 0 else img.filter(ImageFilter.GaussianBlur(radius))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
