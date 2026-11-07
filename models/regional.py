"""Compositional mode: training-free regional prompting (MultiDiffusion-style noise blending).
Each object gets its own text encoding and its own canvas region; the full prompt keeps the scene coherent.
No gradients -> same kind of compute as normal generation, so it is phone-compatible (cost ~ (regions+2)/2 x).
Owner: P1. Does NOT change any evaluation config."""
import re, time, torch
from models.pipeline import DEVICE

RELS = [("to the left of", "L"), ("left of", "L"), ("to the right of", "R"), ("right of", "R"),
        ("next to", "S"), ("beside", "S"), ("near", "S"), ("and", "S"), ("on top of", "T"),
        ("above", "T"), ("over", "T"), ("on", "T"), ("under", "B"), ("below", "B"), ("beneath", "B")]
COUNTS = {"two": 2, "three": 3, "four": 4}

def _singular(phrase):
    w = phrase.split()
    for i, x in enumerate(w):                              # first plural-looking noun -> singular (crude)
        if x.endswith("s") and len(x) > 3 and not x.endswith(("ss", "us", "is")):
            w[i] = x[:-3] + "y" if x.endswith("ies") else x[:-1]; break
    return " ".join(w)

def parse_layout(prompt):
    """Returns (regions, axis): regions = list of region prompts in canvas order; axis 'col' or 'row'. [] = no layout."""
    p = " " + prompt.lower().strip().rstrip(".") + " "
    m = None   # counting disabled: narrow regions merged into one object in the visual test
    if m:                                                  # counting: N copies side by side
        n, rest = COUNTS[m.group(1)], m.group(2)
        noun_phrase = re.split(r"\s(?:on|in|at|under|beside|next to|near|with)\s", rest)[0]
        return [f"a single {_singular(noun_phrase)}"] * n, "col"
    for kw, rel in RELS:                                   # first relation word splits the prompt into A and B
        i = p.find(f" {kw} ")
        if i > 0:
            a, b = p[:i].strip(), p[i + len(kw) + 2:].strip()
            if len(a.split()) < 2 or len(b.split()) < 2:
                continue
            if rel in ("L", "S"): return [a, b], "col"
            if rel == "R":        return [b, a], "col"
            if rel in ("T", "B"): return [], None   # on/under: objects overlap, split hurt in the visual test
    return [], None

def region_masks(n, axis, h, w, feather=0.12):
    """n soft masks (sum to 1 everywhere) splitting the latent canvas into columns or rows."""
    L = w if axis == "col" else h
    x = torch.linspace(0, 1, L)
    centers = [(k + 0.5) / n for k in range(n)]
    raw = torch.stack([torch.clamp(1 - (torch.abs(x - c) - 0.5 / n) / feather, 0, 1) for c in centers])
    raw = raw / raw.sum(0, keepdim=True).clamp(min=1e-6)
    return [(r[None, :].expand(h, w) if axis == "col" else r[:, None].expand(h, w)) for r in raw]

@torch.no_grad()
def regional_generate(pipe, prompt, seed, res=512, steps=None, guidance=None, negative=""):
    cfg = pipe.p3_cfg
    steps, guidance = steps or cfg["steps"], guidance or cfg["guidance"]
    regions, axis = parse_layout(prompt)
    if not regions:
        return None, 0.0, dict(layout=None)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts = [prompt] + regions + [negative]                # global, regions..., unconditional
    tok = pipe.tokenizer(texts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    emb = pipe.text_encoder(tok)[0].to(pipe.unet.dtype)
    h = w = res // pipe.vae_scale_factor
    g = torch.Generator("cpu").manual_seed(int(seed))
    latents = torch.randn((1, pipe.unet.config.in_channels, h, w), generator=g).to(DEVICE, pipe.unet.dtype)
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=DEVICE)
    latents = latents * sched.init_noise_sigma
    masks = [m.to(DEVICE, pipe.unet.dtype) for m in region_masks(len(regions), axis, h, w)]
    n = len(texts)
    for i, t in enumerate(sched.timesteps):
        w_glob = 0.15 if i < 0.4 * steps else 0.4          # early steps: regions set the layout; later: blend for coherence
        x = sched.scale_model_input(latents, t).repeat(n, 1, 1, 1)
        eps = pipe.unet(x, t, encoder_hidden_states=emb).sample
        eps_g, eps_r, eps_u = eps[0:1], eps[1:-1], eps[-1:]
        regional = sum(m[None, None] * eps_r[k:k + 1] for k, m in enumerate(masks))
        cond = w_glob * eps_g + (1 - w_glob) * regional
        latents = sched.step(eps_u + guidance * (cond - eps_u), t, latents).prev_sample
    img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    nsfw = None
    if getattr(pipe, "safety_checker", None) is not None:          # same safety filter as the normal path
        img, nsfw = pipe.run_safety_checker(img, DEVICE, emb.dtype)
    denorm = [not nsfw[0]] if nsfw is not None else None
    img = pipe.image_processor.postprocess(img.float(), output_type="pil", do_denormalize=denorm)[0]
    if DEVICE == "cuda": torch.cuda.synchronize()
    return img, time.perf_counter() - t0, dict(layout=axis, regions=regions)
