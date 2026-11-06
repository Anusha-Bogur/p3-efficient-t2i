"""CLIP score (ViT-L/14, 0-100 scale, same formula as torchmetrics) and clean-FID."""
import torch
from PIL import Image

def clip_score(image_paths, prompts, device="cuda", batch=16):
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    scores = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            ims = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
            inp = proc(text=prompts[i:i + batch], images=ims, return_tensors="pt",
                       padding=True, truncation=True).to(device)
            out = model(**inp)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            scores += (100 * (img * txt).sum(-1)).clamp(min=0).tolist()
    return round(sum(scores) / len(scores), 3)

def fid(gen_dir, ref_dir="data/ref"):
    from cleanfid import fid as cfid
    return round(float(cfid.compute_fid(gen_dir, ref_dir, mode="clean", num_workers=0)), 2)
