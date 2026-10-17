import base64
import hashlib
import io
import os
 
import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
 
PROMPT_STYLES = {
    "Textbook": {
        "ADI": "an H&E image of adipose tissue",
        "BACK": "an H&E image of background with no tissue",
        "DEB": "an H&E image of debris and necrosis",
        "LYM": "an H&E image of lymphocytes",
        "MUC": "an H&E image of mucus",
        "MUS": "an H&E image of smooth muscle",
        "NORM": "an H&E image of normal colon mucosa",
        "STR": "an H&E image of cancer-associated stroma",
        "TUM": "an H&E image of colorectal adenocarcinoma epithelium",
    },
    "Report phrasing": {
        "ADI": "mature adipose tissue is present",
        "BACK": "no tissue identified on this field",
        "DEB": "necrotic debris is seen",
        "LYM": "dense lymphoid infiltrate",
        "MUC": "pools of extracellular mucin",
        "MUS": "bundles of smooth muscle (muscularis propria)",
        "NORM": "benign colonic mucosa without dysplasia",
        "STR": "desmoplastic stroma",
        "TUM": "invasive adenocarcinoma, moderately differentiated",
    },
    "Abbreviation": {
        "ADI": "adipose",
        "BACK": "bkg",
        "DEB": "necrosis",
        "LYM": "lymphs",
        "MUC": "mucin",
        "MUS": "MP",
        "NORM": "nl mucosa",
        "STR": "stroma",
        "TUM": "CRC / adenoCA",
    },
    "Hedged": {
        "ADI": "findings suggestive of adipose tissue",
        "BACK": "possibly no tissue present",
        "DEB": "features suggestive of necrosis",
        "LYM": "likely a lymphoid infiltrate",
        "MUC": "probable mucin",
        "MUS": "features suggestive of smooth muscle",
        "NORM": "probably benign colonic mucosa",
        "STR": "suggestive of reactive stroma",
        "TUM": "suspicious for adenocarcinoma",
    },
}
LABELS = list(PROMPT_STYLES["Textbook"].keys())
MODELS = ["CONCH", "MUSK", "QUILTNET", "PLIP"]
MODE = os.environ.get("DEMO_MODE", "mock").lower()
 

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
 

def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
 
 
def _mock_scores(img: Image.Image, model: str, prompts: dict) -> dict:
    small = np.asarray(img.convert("RGB").resize((16, 16))).astype(np.int32) // 24
    key = small.tobytes() + model.encode() + "".join(prompts.values()).encode()
    rng = np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))
    logits = rng.normal(0, 1, len(prompts))
    logits[0] += 1.5  # give every model a "preferred" class so mock looks plausible
    p = np.exp(logits) / np.exp(logits).sum()
    return dict(zip(prompts.keys(), p.round(3).tolist()))
 
 
def available_models() -> list:
    if MODE != "real":
        return MODELS
    return [m for m in MODELS if os.environ.get(f"ENDPOINT_{m}")]
 
 
def classify(images: list, model: str, prompts: dict) -> list:
    if MODE != "real":
        return [_mock_scores(im, model, prompts) for im in images]
    import requests
    r = requests.post(
        os.environ[f"ENDPOINT_{model}"].rstrip("/") + "/classify",
        headers={"Authorization": f"Bearer {os.environ[f'API_KEY_{model}']}"},
        json={"images": [_b64(im) for im in images], "prompts": prompts},
        timeout=120,
    )
    r.raise_for_status()
    # serve.py returns only the top label: {"results": [{"predicted_label", "score"}]}
    return [{res["predicted_label"]: res["score"]} for res in r.json()["results"]]
 
 
def top(scores: dict):
    label = max(scores, key=scores.get)
    return label, scores[label]
 

def run_image_test(img, stain, blur_r, quality):
    if img is None:
        raise gr.Error("Upload an H&E tile first.")
    img = img.convert("RGB")
    variants = {
        "Original": img,
        "Stain shift": stain_shift(img, stain),
        "Blur": blur(img, blur_r),
        "JPEG": jpeg(img, quality),
        "All three": jpeg(blur(stain_shift(img, stain), blur_r), quality),
    }
    names, ims = list(variants), list(variants.values())
    prompts = PROMPT_STYLES["Textbook"]
 
    rows, flips = [], 0
    for m in available_models():
        preds = [top(s) for s in classify(ims, m, prompts)]
        base = preds[0][0]
        changed = sum(p[0] != base for p in preds[1:])
        flips += changed
        rows.append([m] + [f"{lab} ({sc:.2f})" + (" ⚠" if lab != base else "")
                           for lab, sc in preds] + [f"{changed}/{len(preds) - 1}"])
 
    total = len(available_models()) * (len(names) - 1)
    verdict = ("✅ Stable: no model changed its answer under these shifts."
               if flips == 0 else
               f"⚠ Unstable: {flips} of {total} predictions changed. "
               "Recommend second review before relying on the model.")
    gallery = [(im, n) for n, im in variants.items()]
    return gallery, rows, verdict
 
def run_language_test(img):
    if img is None:
        raise gr.Error("Upload an H&E tile first.")
    img = img.convert("RGB")
    styles = list(PROMPT_STYLES)
    rows, unstable = [], 0
    for m in available_models():
        preds = [top(classify([img], m, PROMPT_STYLES[s])[0]) for s in styles]
        base = preds[0][0]
        changed = sum(p[0] != base for p in preds[1:])
        unstable += changed > 0
        rows.append([m] + [f"{lab} ({sc:.2f})" + (" ⚠" if lab != base else "")
                           for lab, sc in preds])
    verdict = ("✅ Wording-robust: every model gave the same answer for all phrasings."
               if unstable == 0 else
               f"⚠ {unstable} model(s) changed their answer when only the wording changed.")
    return rows, verdict
 
 
with gr.Blocks(title="Pathology Reliability Checker") as demo:
    gr.Markdown(
        "# Pathology Reliability Checker\n"
        "Do pathology vision-language models **see** and **read** like pathologists? "
        f"Mode: **{MODE.upper()}** · Models: {', '.join(available_models()) or 'none configured'}"
    )
    with gr.Tab("Image test"):
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="H&E tile")
                stain = gr.Slider(0, 0.5, value=0.2, step=0.05, label="Stain shift strength")
                blur_r = gr.Slider(0, 4, value=1.5, step=0.5, label="Blur radius (px)")
                quality = gr.Slider(10, 100, value=30, step=5, label="JPEG quality")
                go1 = gr.Button("Run image test", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="What each model sees", columns=5, height=220)
                verdict1 = gr.Markdown()
                table1 = gr.Dataframe(
                    headers=["Model", "Original", "Stain shift", "Blur", "JPEG", "All three", "Changed"],
                    interactive=False)
        go1.click(run_image_test, [img_in, stain, blur_r, quality], [gallery, table1, verdict1])
 
    with gr.Tab("Language test"):
        with gr.Row():
            with gr.Column(scale=1):
                img_in2 = gr.Image(type="pil", label="H&E tile")
                go2 = gr.Button("Run language test", variant="primary")
            with gr.Column(scale=2):
                verdict2 = gr.Markdown()
                table2 = gr.Dataframe(headers=["Model"] + list(PROMPT_STYLES), interactive=False)
        go2.click(run_language_test, [img_in2], [table2, verdict2])
 
if __name__ == "__main__":
    demo.launch()