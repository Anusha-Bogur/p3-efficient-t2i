"""P2. Quality vs cost plot from results/summary.csv -> results/quality_vs_cost.png"""
import csv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [r for r in csv.DictReader(open("results/summary.csv")) if r.get("clip_L14")]
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in rows:
    x, y = float(r["latency_median_s"]), float(r["clip_L14"])
    ax.scatter(x, y, s=60); ax.annotate(r["config"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Median latency per image (s)"); ax.set_ylabel("CLIP score (ViT-L/14)")
ax.set_title("Quality vs cost (same 300 prompts and seeds)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/quality_vs_cost.png", dpi=180); print("saved results/quality_vs_cost.png")
