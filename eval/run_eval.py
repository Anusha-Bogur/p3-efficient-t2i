"""P2 runs this (P1 can too). One configuration -> images + metrics.
Usage: python eval/run_eval.py --config bk_final --n 300
       python eval/run_eval.py --config bk_final --n 5 --tag laptop --no-quality   (laptop proxy timing)"""
import sys, os, json, argparse, statistics, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.pipeline import load_pipeline, generate, model_size, CONFIGS, TFLOPS, DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=list(CONFIGS))
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--tag", default="gpu")
ap.add_argument("--no-quality", action="store_true", help="skip CLIP/FID (timing-only runs)")
a = ap.parse_args()

prompts = [json.loads(l) for l in open("data/prompts.jsonl")][:a.n]
out_dir = f"results/{a.config}_{a.tag}"; os.makedirs(out_dir, exist_ok=True)
pipe = load_pipeline(a.config)
for _ in range(2): generate(pipe, "warm up", 0, a.res)            # warm-up, not timed
if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()

times, paths = [], []
for i, p in enumerate(prompts):
    img, t = generate(pipe, p["prompt"], seed=p["id"], res=a.res)
    path = f"{out_dir}/{p['id']}.png"; img.save(path); paths.append(path); times.append(t)
    if i % 25 == 0: print(f"{a.config}: {i+1}/{len(prompts)}  last {t:.2f}s", flush=True)

row = dict(config=a.config, tag=a.tag, device=DEVICE, n=len(prompts), res=a.res,
           steps=CONFIGS[a.config]["steps"], lora_loaded=getattr(pipe, "p3_lora", None),
           latency_median_s=round(statistics.median(times), 3),
           latency_p95_s=round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
           peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None,
           model_MB=model_size(pipe)["total_MB"], tflops_per_image=TFLOPS[a.config])
if DEVICE == "cuda":
    row["gpu"] = torch.cuda.get_device_name(0)
del pipe; torch.cuda.empty_cache() if DEVICE == "cuda" else None
if not a.no_quality:
    from eval.metrics import clip_score, fid
    row["clip_L14"] = clip_score(paths, [p["prompt"] for p in prompts], device=DEVICE if DEVICE != "mps" else "cpu")
    row[f"fid_{len(prompts)}"] = fid(out_dir)
json.dump(row, open(f"{out_dir}/metrics.json", "w"), indent=1)
new = not os.path.exists("results/summary.csv")
with open("results/summary.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); 
    if new: w.writeheader()
    w.writerow(row)
print(json.dumps(row, indent=1))
