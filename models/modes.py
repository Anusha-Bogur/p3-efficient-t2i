"""Generation modes (P1). Does NOT change any evaluation config.
Shipped in the app:
  Low           = standard bk_final generation: TAESD, 20 DPM-Solver++ steps (~1.9-2.2 s on a T4)
  Compositional = regional prompting for side-by-side prompts: one text encoding + canvas region per object,
                  Low settings otherwise (~3.5 s); other prompts fall back to Low
  Medium        = 3 candidates (seeds s, s+1, s+2) at 25 steps decoded with the full SD VAE;
                  CLIP ViT-B/32 picks the best prompt match (~7.8 s)
Experimental, server-only, NOT exposed in the app (need gradients or extra detector models; not evaluated on the full model):
  High          = Attend-and-Excite
  Verified      = High + OWLv2 object check + CLIP colour check, up to 3 retries
"""
import re, time, torch
from diffusers import StableDiffusionAttendAndExcitePipeline
from models.pipeline import generate, DEVICE

_NLP, _RANK, _DET = None, None, None
STOP = {"a", "an", "the", "of", "on", "in", "with", "and", "to", "left", "right", "beside", "next", "near",
        "behind", "above", "below", "under", "top", "front", "is", "are", "two", "three", "four", "some"}

def _nouns(prompt):
    global _NLP
    try:
        import spacy
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm")
        return [t.text.lower() for t in _NLP(prompt) if t.pos_ in ("NOUN", "PROPN")]
    except Exception:                       # fallback if spaCy is missing: last two content words
        return [w for w in re.findall(r"[a-z]+", prompt.lower()) if w not in STOP][-2:]

def token_indices(pipe, prompt, max_n=4):
    nouns = set(_nouns(prompt))
    toks = pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(prompt).input_ids)
    return [i for i, t in enumerate(toks) if t.replace("</w>", "") in nouns][:max_n]

QUALITY_STEPS = 25          # Medium/High/Verified use more steps + the full VAE for finer detail

def make_ae(pipe):
    """Shares every weight with the main pipeline (LoRA included).
    Also loads the full SD VAE once (~0.17 GB in fp16) so Medium can decode with it (Low/Compositional keep TAESD)."""
    from diffusers import AutoencoderKL
    pipe.p3_taesd = pipe.vae
    pipe.p3_fullvae = AutoencoderKL.from_pretrained(pipe.p3_cfg["model"], subfolder="vae",
                                                    torch_dtype=pipe.unet.dtype).to(DEVICE)
    ae = StableDiffusionAttendAndExcitePipeline(**{k: v for k, v in pipe.components.items() if k != "image_encoder"})
    ae.set_progress_bar_config(disable=True)
    return ae

def _quality(pipe, ae, on):
    """Low: TAESD + config steps (identical to the evaluated bk_final). Others: full VAE + QUALITY_STEPS."""
    vae = pipe.p3_fullvae if on else pipe.p3_taesd
    pipe.vae = vae; ae.vae = vae
    if not hasattr(pipe, "p3_base_steps"):
        pipe.p3_base_steps = pipe.p3_cfg["steps"]
    pipe.p3_cfg = dict(pipe.p3_cfg, steps=QUALITY_STEPS if on else pipe.p3_base_steps)

def _ranker():
    global _RANK
    if _RANK is None:
        from transformers import CLIPModel, CLIPProcessor
        _RANK = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval(),
                 CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    return _RANK

def best_of(pipe, prompt, seed, n=3):
    model, proc = _ranker()
    cands = [generate(pipe, prompt, seed + i) for i in range(n)]
    imgs = [c[0] for c in cands]
    with torch.no_grad():
        inp = proc(text=[prompt], images=imgs, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        s = model(**inp).logits_per_image.squeeze(1).float().tolist()
    k = max(range(n), key=lambda i: s[i])
    return imgs[k], sum(c[1] for c in cands), dict(candidate_clip_B32=[round(x, 2) for x in s], picked=k)

def attend_excite(pipe, ae, prompt, seed):
    idx = token_indices(pipe, prompt)
    if not idx:                              # no nouns found: plain generation
        img, t = generate(pipe, prompt, seed); return img, t, dict(token_indices=[])
    cfg = pipe.p3_cfg; steps = cfg["steps"]
    g = torch.Generator("cpu").manual_seed(int(seed))
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = ae(prompt, token_indices=idx, num_inference_steps=steps, guidance_scale=cfg["guidance"], generator=g,
             max_iter_to_alter=steps // 2, thresholds={0: 0.05, steps // 4: 0.5, steps // 2: 0.8},
             attn_res=(16, 16)).images[0]
    if DEVICE == "cuda": torch.cuda.synchronize()
    words = pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(prompt).input_ids)
    return img, time.perf_counter() - t0, dict(attended_words=[words[i].replace("</w>", "") for i in idx])

def _detector():
    global _DET
    if _DET is None:
        from transformers import pipeline as hf_pipeline
        _DET = hf_pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble",
                           device=0 if DEVICE == "cuda" else -1)
    return _DET

COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
           "black", "white", "gray", "silver", "gold"]

def colour_pairs(prompt):
    """{noun: colour} for phrases like 'a red car' (spaCy if available, else regex)."""
    try:
        import spacy
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm")
        out = {}
        for t in _NLP(prompt):
            if t.pos_ in ("NOUN", "PROPN"):
                for c in t.children:
                    if c.dep_ == "amod" and c.text.lower() in COLOURS:
                        out[t.text.lower()] = c.text.lower()
        return out
    except Exception:
        return {n.lower(): c.lower() for c, n in re.findall(r"\b(" + "|".join(COLOURS) + r")\s+([a-z]+)", prompt.lower())}

def check_image(img, prompt, thr=0.2):
    """Object presence (OWLv2) + colour of each coloured object (CLIP B/32 on the detected crop)."""
    nouns = sorted(set(_nouns(prompt)))
    want = colour_pairs(prompt)
    if not nouns:
        return dict(ok=True, missing=[], wrong_colour={}, found=[])
    dets = [d for d in _detector()(img, candidate_labels=nouns) if d["score"] >= thr]
    best = {}
    for d in dets:
        if d["label"] not in best or d["score"] > best[d["label"]]["score"]:
            best[d["label"]] = d
    missing = [n for n in nouns if n not in best]
    wrong = {}
    if want:
        model, proc = _ranker()
        for noun, colour in want.items():
            if noun not in best:
                continue
            b = best[noun]["box"]
            crop = img.crop((b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
            texts = [f"a photo of a {c} {noun}" for c in COLOURS]
            with torch.no_grad():
                inp = proc(text=texts, images=[crop], return_tensors="pt", padding=True).to(DEVICE)
                got = COLOURS[int(model(**inp).logits_per_image[0].argmax())]
            if got != colour:
                wrong[noun] = f"wanted {colour}, got {got}"
    return dict(ok=not missing and not wrong, missing=missing, wrong_colour=wrong, found=sorted(best))

def objects_found(img, prompt, thr=0.2):
    """Checks object PRESENCE only (not colours, counts, positions or actions)."""
    nouns = sorted(set(_nouns(prompt)))
    if not nouns:
        return True, [], []
    found = {d["label"] for d in _detector()(img, candidate_labels=nouns) if d["score"] >= thr}
    missing = [n for n in nouns if n not in found]
    return not missing, sorted(found), missing

def verified(pipe, ae, prompt, seed, tries=3):
    t0, best = time.perf_counter(), None
    for k in range(tries):
        img, _, info = attend_excite(pipe, ae, prompt, seed + k)
        r = check_image(img, prompt)
        errors = len(r["missing"]) + len(r["wrong_colour"])
        if best is None or errors < best[2]:
            best = (img, k, errors, r)
        if r["ok"]:
            return img, time.perf_counter() - t0, dict(verified=True, status="verified: all objects and colours correct",
                                                      attempt=k + 1, **r)
    img, k, _, r = best
    return img, time.perf_counter() - t0, dict(verified=False, status=f"best attempt after {tries} tries",
                                              attempt=k + 1, **r)

def compositional(pipe, prompt, seed):
    from models.regional import regional_generate
    img, t, info = regional_generate(pipe, prompt, seed)
    if img is None:                                   # single-object prompt: normal generation
        img, t = generate(pipe, prompt, seed); info = dict(layout=None, note="no multi-object layout found")
    return img, t, info

def generate_mode(pipe, ae, prompt, seed, mode="Low"):
    if mode == "Compositional":
        _quality(pipe, ae, False)          # Compositional uses Low settings (TAESD, 20 steps), as evaluated
        return compositional(pipe, prompt, seed)
    _quality(pipe, ae, mode != "Low")
    if mode == "Verified": return verified(pipe, ae, prompt, seed)
    if mode == "Medium": return best_of(pipe, prompt, seed)
    if mode == "High":   return attend_excite(pipe, ae, prompt, seed)
    img, t = generate(pipe, prompt, seed); return img, t, {}
