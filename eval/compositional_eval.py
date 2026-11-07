"""P1 (GPU 1): does Medium/High give more CORRECT images than Low on multi-object prompts?
Scores: CLIP ViT-L/14 (0-100), object presence (OWLv2) and objects+colours correct (OWLv2 + CLIP-B/32 crop check).
Note: Verified mode uses the same checker, so its presence/colour % is optimistic; CLIP-L/14 is the independent score.
Usage: CUDA_VISIBLE_DEVICES=1 python eval/compositional_eval.py --config bk_final  (or bk_dpm20_taesd)"""
import sys, os, json, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from models.pipeline import load_pipeline, DEVICE
from models.modes import make_ae, generate_mode, check_image
from eval.metrics import clip_score
from eval.prompts_comp import PROMPTS
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="bk_final"); ap.add_argument("--modes", default="Low,Compositional,Medium,High,Verified")
a = ap.parse_args()
pipe = load_pipeline(a.config); ae = make_ae(pipe)
report = {}
for mode in a.modes.split(","):
    out = f"results/compositional/{mode}"; os.makedirs(out, exist_ok=True)
    paths, times, present, correct = [], [], [], []
    for i, p in enumerate(PROMPTS):
        img, t, _ = generate_mode(pipe, ae, p, 1000 + i, mode)
        path = f"{out}/{i:02d}.png"; img.save(path); paths.append(path); times.append(t)
        r = check_image(img, p)
        present.append(not r["missing"]); correct.append(r["ok"])
    report[mode] = dict(clip_L14=clip_score(paths, PROMPTS, device=DEVICE),
                        all_objects_present_pct=round(100 * sum(present) / len(present), 1),
                        objects_and_colours_correct_pct=round(100 * sum(correct) / len(correct), 1),
                        latency_median_s=round(statistics.median(times), 2))
    print(mode, report[mode], flush=True)
json.dump(report, open("results/compositional/report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
