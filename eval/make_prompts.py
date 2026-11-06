"""P2. Fixed evaluation set: N COCO val2017 captions (one per image, seed 2026) + their real images (FID reference).
Needs only data/coco/annotations/captions_val2017.json - images are downloaded by URL.
Usage: python eval/make_prompts.py --n 300"""
import json, random, argparse, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--ann", default="data/coco/annotations/captions_val2017.json")
ap.add_argument("--n", type=int, default=300); a = ap.parse_args()
caps = json.load(open(a.ann))
first = {}
for c in caps["annotations"]:
    first.setdefault(c["image_id"], c["caption"].strip())
info = {im["id"]: im for im in caps["images"]}
ids = random.Random(2026).sample(sorted(first), a.n)
os.makedirs("data/ref", exist_ok=True)
def get(i):
    dst = f"data/ref/{info[i]['file_name']}"
    if not os.path.exists(dst): urllib.request.urlretrieve(info[i]["coco_url"], dst)
with ThreadPoolExecutor(16) as ex: list(ex.map(get, ids))
with open("data/prompts.jsonl", "w") as f:
    for i in ids: f.write(json.dumps({"id": i, "prompt": first[i], "file": info[i]["file_name"]}) + "\n")
print(f"wrote data/prompts.jsonl + data/ref/ ({a.n} images)")
