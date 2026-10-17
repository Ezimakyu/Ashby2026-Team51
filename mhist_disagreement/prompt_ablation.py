cat > /projects/bfqi/teams/team51/prompt_ablation.py << 'EOF'
"""Prompt ablation: compare zero-shot AUC of prompt sets on unanimous MHIST images.

Usage (on Delta):
  export EP="http://<server>:<port>"   # model server endpoint (see #compute-issues)
  export KEY="<api key>"               # never commit keys
  python3 prompt_ablation.py <mhist_dir> <prompt_sets.json> [n_per_class]
"""
import json, base64, os, sys, csv, random, requests

mhist_dir, prompt_file = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
ep, key = os.environ["EP"], os.environ["KEY"]

sets = json.load(open(prompt_file))
rows = list(csv.reader(open(os.path.join(mhist_dir, "annotations.csv"))))[1:]
random.seed(0)
hp  = random.sample([r[0] for r in rows if r[2] == "0"], n)   # 0/7 chose SSA
ssa = random.sample([r[0] for r in rows if r[2] == "7"], n)   # 7/7 chose SSA
names, y = hp + ssa, [0] * n + [1] * n
b64 = [base64.b64encode(open(os.path.join(mhist_dir, "images", f), "rb").read()).decode() for f in names]

def p_ssa(prompts):
    r = requests.post(ep + "/classify", headers={"Authorization": "Bearer " + key},
                      json={"images": b64, "prompts": prompts}, timeout=300).json()["results"]
    # server returns only the top label + score; for binary, P(SSA) = score or 1 - score
    return [x["score"] if x["predicted_label"] == "SSA" else 1 - x["score"] for x in r]

def auc(y, s):
    pos = [b for a, b in zip(y, s) if a == 1]; neg = [b for a, b in zip(y, s) if a == 0]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

print(f"endpoint={ep}  n={n} per class  seed=0")
for name, pr in sets.items():
    if isinstance(pr["HP"], list):   # multiple prompts per class: average P(SSA)
        runs = [p_ssa({"HP": h, "SSA": s}) for h, s in zip(pr["HP"], pr["SSA"])]
        scores = [sum(v) / len(v) for v in zip(*runs)]
    else:
        scores = p_ssa(pr)
    print(f"{name:16s} AUC = {auc(y, scores):.3f}")
EOF
