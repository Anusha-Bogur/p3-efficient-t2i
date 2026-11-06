"""P2. CLIP score (ViT-L/14, 0-100 scale) and clean-FID."""
import numpy as np, torch
from PIL import Image

def clip_score(image_paths, prompts, device="cuda", batch=16):
    from torchmetrics.multimodal.clip_score import CLIPScore
    m = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    for i in range(0, len(image_paths), batch):
        ims = [torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((512, 512)))).permute(2, 0, 1)
               for p in image_paths[i:i + batch]]
        m.update(torch.stack(ims).to(device), [t[:300] for t in prompts[i:i + batch]])
    return round(float(m.compute()), 3)

def fid(gen_dir, ref_dir="data/ref"):
    from cleanfid import fid as cfid
    return round(float(cfid.compute_fid(gen_dir, ref_dir, mode="clean", num_workers=0)), 2)
