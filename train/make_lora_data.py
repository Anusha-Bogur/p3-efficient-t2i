"""P1. 1,000 COCO *train* images + captions -> data/coco_lora (imagefolder with metadata.jsonl).
Train split only: never overlaps the val prompts used for evaluation.
Usage: python train/make_lora_data.py --n 1000"""
import json, random, argparse, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--ann", default="data/coco/annotations/captions_train2017.json")
ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
caps = json.load(open(a.ann))
first = {}
for c in caps["annotations"]:
    first.setdefault(c["image_id"], c["caption"].strip())
info = {im["id"]: im for im in caps["images"]}
ids = random.Random(7).sample(sorted(first), a.n)
out = "data/coco_lora"; os.makedirs(out, exist_ok=True)
def get(i):
    dst = f"{out}/{info[i]['file_name']}"
    if not os.path.exists(dst): urllib.request.urlretrieve(info[i]["coco_url"], dst)
with ThreadPoolExecutor(16) as ex: list(ex.map(get, ids))
with open(f"{out}/metadata.jsonl", "w") as f:
    for i in ids: f.write(json.dumps({"file_name": info[i]["file_name"], "text": first[i]}) + "\n")
print(f"wrote {out} ({a.n} images)")
