import csv, sys
from pathlib import Path
import numpy as np
from hub.audio import load_clip
from hub.classifier import make_classifier
from hub.config import load_config

NEG = ["door_wood_knock", "can_opening", "clapping", "footsteps",
       "door_wood_creaks", "keyboard_typing", "coughing", "dog"]

def peak_db(x):
    b = x[: len(x) // 512 * 512].reshape(-1, 512)
    return float((20 * np.log10(np.sqrt((b ** 2).mean(axis=1)) + 1e-9)).max())

esc = Path(sys.argv[1])
clf = make_classifier(load_config())
with open(esc / "meta" / "esc50.csv", newline="") as f:
    rows = [r for r in csv.DictReader(f)
            if r["fold"] != "5" and r["category"] in NEG + ["glass_breaking"]]
s = {"yamnet": {"g": [], "n": []}, "energy": {"g": [], "n": []}}
for i, r in enumerate(rows, 1):
    x = load_clip((esc / "audio" / r["filename"]).read_bytes())
    k = "g" if r["category"] == "glass_breaking" else "n"
    s["yamnet"][k].append(clf.classify(x).glass_score)
    s["energy"][k].append(peak_db(x))
    if i % 50 == 0:
        print(f"  {i}/{len(rows)} clips")
print(f"\nFolds 1-4: {len(s['yamnet']['g'])} glass, {len(s['yamnet']['n'])} negative clips")
for name in ("energy", "yamnet"):
    g, n = np.array(s[name]["g"]), np.array(s[name]["n"])
    thr = np.quantile(g, 0.10, method="lower")
    tp, fp = int((g >= thr).sum()), int((n >= thr).sum())
    rec, prec = tp / len(g), tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"{name:7} threshold {thr:8.3f}  recall {rec:.2f}  precision {prec:.2f}  "
          f"F1 {f1:.2f}  false-positive rate {fp / len(n):.2f}")
