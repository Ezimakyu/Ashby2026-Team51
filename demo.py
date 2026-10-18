"""Pathology Reliability Checker: an HP vs SSA second-opinion demo on MHIST.

Tabs:
    Second opinion  upload a tile -> MUSK /embed -> p(SSA) from the soft head,
                    a "recommend second review" flag from the conflict head,
                    the 5 most similar training tiles with their votes from the
                    7 pathologists, and an optional one-line report-style summary
                    written by a Lumen LLM
    Image test      does each model's call change under stain shift, blur or JPEG?
    Language test   does it change when only the prompt wording changes?

DEMO_MODE=mock (the default) fakes every model call, so the UI runs anywhere.
DEMO_MODE=real calls the hosted servers. They only answer from inside Delta,
so run it in an OnDemand Desktop session and open the localhost link there.
Real mode reads:
    ENDPOINT_<MODEL>, API_KEY_<MODEL>   per hosted model; Second opinion needs MUSK
    HEADS_PATH      heads.py output (default mhist_disagreement/models/musk_heads.joblib)
    MHIST_IMAGES    MHIST images folder, to show the similar training cases
    LUMEN_API_KEY   optional, for the report line; LUMEN_MODEL picks the model
MHIST's terms don't allow redistributing its images: never launch with
share=True (that makes a public link).
"""

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mhist_disagreement"))
from perturb import blur, jpeg, stain_shift  # noqa: E402
import hosted  # noqa: E402

# HP vs SSA prompt styles, from mhist_disagreement/prompt_sets.json (styles 1-5;
# 6_observed is left out because it gives several prompts per class).
PROMPT_STYLES = {
    "Name + morphology": {
        "HP": "a histopathology image of a hyperplastic polyp of the colon, with serration limited to the upper crypts and narrow, straight crypt bases",
        "SSA": "a histopathology image of a sessile serrated adenoma of the colon, with serration extending to the crypt base and dilated, boot-shaped crypts",
    },
    "Name only": {
        "HP": "an H&E image of a hyperplastic polyp",
        "SSA": "an H&E image of a sessile serrated adenoma",
    },
    "Morphology only": {
        "HP": "colonic mucosa with straight, evenly spaced crypts that are serrated only in the upper half, with narrow crypt bases",
        "SSA": "colonic mucosa with crypts serrated down to the base, with dilated crypt bases spreading sideways into L-shaped or boot-shaped crypts",
    },
    "New name (SSL)": {
        "HP": "an H&E image of a hyperplastic polyp of the colon",
        "SSA": "an H&E image of a sessile serrated lesion of the colon",
    },
    "Report style": {
        "HP": "Colon, polyp, biopsy: hyperplastic polyp.",
        "SSA": "Colon, polyp, biopsy: sessile serrated adenoma.",
    },
}
DEFAULT_STYLE = "Name + morphology"
LABELS = list(PROMPT_STYLES[DEFAULT_STYLE].keys())

# Morphology concepts scored zero-shot (present vs absent prompt) for the report line.
CONCEPTS = {
    "serration down to the crypt base": {
        "present": "colonic crypts serrated all the way down to the crypt base",
        "absent": "colonic crypts serrated only in the upper half, with smooth crypt bases",
    },
    "dilated, boot-shaped crypt bases": {
        "present": "colonic crypts with dilated bases spreading sideways into L-shaped or boot-shaped crypts",
        "absent": "colonic crypts with narrow, straight crypt bases",
    },
    "horizontal crypt growth": {
        "present": "crypts growing horizontally along the muscularis mucosae",
        "absent": "crypts oriented vertically, perpendicular to the muscularis mucosae",
    },
}

MODELS = ["CONCH", "MUSK", "QUILTNET", "PLIP"]
SECOND_OPINION_MODEL = "MUSK"
MODE = os.environ.get("DEMO_MODE", "mock").lower()
HEADS_PATH = os.environ.get("HEADS_PATH", str(HERE / "mhist_disagreement" / "models" / "musk_heads.joblib"))
MHIST_IMAGES = os.environ.get("MHIST_IMAGES", "")
LUMEN_BASE = os.environ.get("LUMEN_BASE_URL", "https://lumen.ncsa.illinois.edu/v1")
LUMEN_MODEL = os.environ.get("LUMEN_MODEL", "ornith-1.0-35b")  # cheapest Lumen model as of Oct 16
N_SIMILAR = 5


def _rng_for(img: Image.Image, *salt: str):
    small = np.asarray(img.convert("RGB").resize((16, 16))).astype(np.int32) // 24
    key = small.tobytes() + "".join(salt).encode()
    return np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))


def _mock_scores(img: Image.Image, model: str, prompts: dict) -> dict:
    rng = _rng_for(img, model, *prompts.values())
    logits = rng.normal(0, 1, len(prompts))
    logits[0] += 1.5  # give every model a "preferred" class so mock looks plausible
    p = np.exp(logits) / np.exp(logits).sum()
    return dict(zip(prompts.keys(), p.round(3).tolist()))


def available_models() -> list:
    if MODE != "real":
        return MODELS
    return [m for m in MODELS if os.environ.get(f"ENDPOINT_{m}")]


def _server(model: str):
    """hosted.post() takes an object with endpoint, api_key and timeout."""
    endpoint = os.environ.get(f"ENDPOINT_{model}")
    if not endpoint:
        raise gr.Error(f"Set ENDPOINT_{model} and API_KEY_{model} for real mode.")
    return type("Server", (), {"endpoint": endpoint.rstrip("/"),
                               "api_key": os.environ.get(f"API_KEY_{model}", ""), "timeout": 120})


def _post(model: str, route: str, payload: dict) -> dict:
    try:
        return hosted.post(_server(model), route, payload)
    except SystemExit as e:  # hosted.post exits on a bad key; show it instead of stopping the app
        raise gr.Error(str(e))


def classify(images: list, model: str, prompts: dict) -> list:
    if MODE != "real":
        return [_mock_scores(im, model, prompts) for im in images]
    result = _post(model, "classify", {"images": [hosted.encode_pil(im) for im in images], "prompts": prompts})
    # serve.py returns only the top label: {"results": [{"predicted_label", "score"}]}
    return [{res["predicted_label"]: res["score"]} for res in result["results"]]


def top(scores: dict):
    label = max(scores, key=scores.get)
    return label, scores[label]


# ---------- Second opinion ----------

@lru_cache(maxsize=1)
def load_heads():
    if not Path(HEADS_PATH).exists():
        raise gr.Error(f"No heads file at {HEADS_PATH}. Build it with mhist_disagreement/heads.py, "
                       "or set HEADS_PATH.")
    import joblib
    return joblib.load(HEADS_PATH)


def concept_scores(img: Image.Image) -> dict:
    """p(present) per concept. With two prompts the server's score is a softmax of
    two cosine similarities, so values sit near 0.5: read them as relative cues."""
    out = {}
    for name, pair in CONCEPTS.items():
        label, score = top(classify([img], SECOND_OPINION_MODEL, pair)[0])
        out[name] = round(score if label == "present" else 1 - score, 3)
    return out


def _vote_tag(k: int) -> str:
    if k in (0, 7):
        return "unanimous"
    return "near-unanimous" if k in (1, 6) else "contested"


def _similar_image(name: str) -> Image.Image:
    path = Path(MHIST_IMAGES) / name
    return Image.open(path).convert("RGB") if MHIST_IMAGES and path.exists() else Image.new("RGB", (224, 224), "#d9d9d9")


def _mock_opinion(img: Image.Image) -> dict:
    rng = _rng_for(img, "second-opinion")
    votes = rng.integers(0, 8, N_SIMILAR)
    return {
        "p_ssa": float(rng.uniform(0.05, 0.95)), "conflict": float(rng.uniform(0, 2.5)),
        "threshold": 1.6, "similar": [(f"mock_{i}.png", int(k), float(rng.uniform(0.8, 0.95)))
                                      for i, k in enumerate(votes)],
    }


def _real_opinion(img: Image.Image) -> dict:
    import heads as heads_mod
    heads = load_heads()
    emb = np.asarray(_post(SECOND_OPINION_MODEL, "embed", {"images": [hosted.encode_pil(img)]})["embeddings"][0],
                     dtype=np.float32)
    pred = heads_mod.predict(heads, emb)
    idx, sims = heads_mod.neighbours(heads, emb, N_SIMILAR)
    return {
        "p_ssa": float(pred["p_ssa"][0]), "conflict": float(pred["conflict"][0]),
        "threshold": heads["flag_threshold"],
        "similar": [(heads["train_image"][i], int(heads["train_votes"][i]), float(s)) for i, s in zip(idx, sims)],
    }


def run_second_opinion(img):
    if img is None:
        raise gr.Error("Upload an H&E tile first.")
    img = img.convert("RGB")
    r = _real_opinion(img) if MODE == "real" else _mock_opinion(img)
    r["flag"] = r["conflict"] >= r["threshold"]
    r["concepts"] = concept_scores(img)
    p = r["p_ssa"]
    n_contested = sum(_vote_tag(k) == "contested" for _, k, _ in r["similar"])

    lean = "SSA" if p >= 0.5 else "HP"
    flag = (f"**⚠ Uncertain: recommend second review.** Predicted disagreement {r['conflict']:.2f} "
            f"(0 = unanimous, 3 = 4:3 split) is in the top 20% of training cases (threshold {r['threshold']:.2f})."
            if r["flag"] else
            f"**✅ Low predicted disagreement** ({r['conflict']:.2f}; second-review threshold {r['threshold']:.2f}).")
    verdict = "\n\n".join([
        f"### P(SSA) = {p:.2f}, leaning {lean}",
        flag,
        f"**Similar training cases:** {n_contested} of {len(r['similar'])} were contested "
        "(2–5 of 7 pathologists voted SSA).",
        "_Research prototype on MHIST, not for diagnosis._" + (" **Mock mode: numbers are fake.**" if MODE != "real" else ""),
    ])
    gallery = [(_similar_image(name), f"{k}/7 SSA · {_vote_tag(k)} · sim {s:.2f}") for name, k, s in r["similar"]]
    concepts = [[name, f"{v:.2f}"] for name, v in r["concepts"].items()]
    return verdict, gallery, concepts, r, ""


def _template_line(r: dict) -> str:
    lean = "sessile serrated lesion" if r["p_ssa"] >= 0.5 else "hyperplastic polyp"
    review = "recommend second review" if r["flag"] else "low predicted disagreement"
    return f"Serrated colonic polyp, model favours {lean} (P(SSA) {r['p_ssa']:.2f}); {review}."


def run_lumen_summary(r):
    if not r:
        raise gr.Error("Run the second opinion first.")
    key = os.environ.get("LUMEN_API_KEY")
    if not key:
        return _template_line(r) + "\n\n_(template; set LUMEN_API_KEY for an LLM-written line)_"
    votes = ", ".join(f"{k}/7 SSA" for _, k, _ in r["similar"])
    concepts = ", ".join(f"{n} {v:.2f}" for n, v in r["concepts"].items())
    facts = (f"P(SSA) from a classifier on MUSK embeddings: {r['p_ssa']:.2f}. "
             f"Predicted pathologist disagreement: {r['conflict']:.2f} on a 0-3 scale; "
             f"second-review flag: {'yes' if r['flag'] else 'no'}. "
             f"Votes on the {len(r['similar'])} most similar training cases: {votes}. "
             f"Zero-shot morphology cue scores (0.5 = neutral, weak signal): {concepts}.")
    import requests
    try:
        resp = requests.post(f"{LUMEN_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={
            "model": LUMEN_MODEL, "temperature": 0.2, "max_tokens": 200,
            "messages": [
                {"role": "system", "content": (
                    "You turn a research model's output on one colon polyp H&E tile into ONE report-style "
                    "sentence for a pathologist. Use only the numbers given. Say which diagnosis the model "
                    "favours (hyperplastic polyp vs sessile serrated lesion), how confident it is, and whether "
                    "a second review is recommended. Don't add findings that weren't given, and don't present "
                    "it as a final diagnosis.")},
                {"role": "user", "content": facts},
            ]}, timeout=60)
        resp.raise_for_status()
        line = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001 - show the reason, fall back to the template
        return _template_line(r) + f"\n\n_(template; Lumen call failed: {e})_"
    return f"{line}\n\n_Written by {LUMEN_MODEL} on Lumen from the numbers above; check it before use._"


# ---------- Image and language tests ----------

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
    prompts = PROMPT_STYLES[DEFAULT_STYLE]

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
        "Hyperplastic polyp vs sessile serrated lesion on MHIST: a second opinion that knows when "
        "pathologists would disagree, and checks of whether models **see** and **read** like pathologists. "
        f"Mode: **{MODE.upper()}** · Models: {', '.join(available_models()) or 'none configured'}"
    )
    with gr.Tab("Second opinion"):
        opinion = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                img_in0 = gr.Image(type="pil", label="H&E tile")
                go0 = gr.Button("Get second opinion", variant="primary")
                concept_table = gr.Dataframe(headers=["Morphology cue (zero-shot)", "p(present)"],
                                             interactive=False)
            with gr.Column(scale=2):
                verdict0 = gr.Markdown()
                similar = gr.Gallery(label="Most similar training cases and their pathologist votes",
                                     columns=N_SIMILAR, height=240)
                go_lumen = gr.Button("Draft report line (Lumen)")
                report = gr.Markdown()
        go0.click(run_second_opinion, [img_in0], [verdict0, similar, concept_table, opinion, report])
        go_lumen.click(run_lumen_summary, [opinion], [report])

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
    demo.launch(share=False)  # never share=True: the demo shows MHIST images
