
"""Zero-shot P(SSA) for every image in one MHIST partition, using one prompt set.

Usage: export EP=... KEY=...

       python3 zero_shot_full.py <mhist_dir> <prompt_sets.json> <set_name> <out.csv> [partition]

"""

import json, base64, os, sys, csv, requests

mhist, pfile, set_name, out = sys.argv[1:5]

part = sys.argv[5] if len(sys.argv) > 5 else "test"

ep, key = os.environ["EP"], os.environ["KEY"]

prompts = json.load(open(pfile))[set_name]

rows = [r for r in list(csv.reader(open(os.path.join(mhist, "annotations.csv"))))[1:] if r[3] == part]



def classify(names, pr):

    b64 = [base64.b64encode(open(os.path.join(mhist, "images", f), "rb").read()).decode() for f in names]

    res = requests.post(ep + "/classify", headers={"Authorization": "Bearer " + key},

                        json={"images": b64, "prompts": pr}, timeout=600).json()["results"]

    return [x["score"] if x["predicted_label"] == "SSA" else 1 - x["score"] for x in res]



def p_ssa(names):

    if isinstance(prompts["HP"], list):   # average over prompt variants

        runs = [classify(names, {"HP": h, "SSA": s}) for h, s in zip(prompts["HP"], prompts["SSA"])]

        return [sum(v) / len(v) for v in zip(*runs)]

    return classify(names, prompts)



names = [r[0] for r in rows]

scores = []

for i in range(0, len(names), 64):          # send in batches of 64

    scores += p_ssa(names[i:i+64])

    print(f"{min(i+64, len(names))}/{len(names)}", flush=True)



with open(out, "w", newline="") as f:

    w = csv.writer(f); w.writerow(["image", "votes_ssa", "majority", "p_ssa"])

    for r, s in zip(rows, scores): w.writerow([r[0], r[2], r[1], f"{s:.6f}"])



def auc(y, s):

    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]

    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

y = [r[1] == "SSA" for r in rows]

votes = [int(r[2]) for r in rows]

unc = [-abs(s - 0.5) for s in scores]                 # closer to 0.5 = more uncertain

split = [v in (3, 4) for v in votes]                  # 4:3 or 3:4 votes

print(f"label AUROC = {auc(y, scores):.3f}")

print(f"disagreement AUROC (uncertainty vs 3-4 votes) = {auc(split, unc):.3f}")

print("saved", out)

