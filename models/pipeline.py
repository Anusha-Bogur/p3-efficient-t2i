"""P1 owns this file. One place that defines every configuration used in the app and the ablation."""
import os, time, torch
from diffusers import (StableDiffusionPipeline, DDIMScheduler,
                       DPMSolverMultistepScheduler, AutoencoderTiny)

DEVICE = os.environ.get("P3_DEVICE") or ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
SD14  = os.environ.get("P3_SD14",  "CompVis/stable-diffusion-v1-4")
BK    = os.environ.get("P3_BK",    "nota-ai/bk-sdm-small-2m")
TAESD = os.environ.get("P3_TAESD", "madebyollin/taesd")
LORA  = os.environ.get("P3_LORA",  "lora_bk")          # folder written by train/lora.sh

# Ablation rows 1-6. Each row changes ONE thing versus the row before it.
CONFIGS = {
    "sd14_fp32_50":   dict(model=SD14, fp16=False, sched="ddim", steps=50, guidance=7.5, taesd=False, lora=False),
    "sd14_fp16_50":   dict(model=SD14, fp16=True,  sched="ddim", steps=50, guidance=7.5, taesd=False, lora=False),
    "bk_fp16_50":     dict(model=BK,   fp16=True,  sched="ddim", steps=50, guidance=7.5, taesd=False, lora=False),
    "bk_dpm20":       dict(model=BK,   fp16=True,  sched="dpm",  steps=20, guidance=7.5, taesd=False, lora=False),
    "bk_dpm20_taesd": dict(model=BK,   fp16=True,  sched="dpm",  steps=20, guidance=7.5, taesd=True,  lora=False),
    "bk_final":       dict(model=BK,   fp16=True,  sched="dpm",  steps=20, guidance=7.5, taesd=True,  lora=True),
}
# Exact compute per 512px image (FLOP counts from the design doc). LoRA is fused, so it adds nothing.
TFLOPS = {"sd14_fp32_50": 82.84, "sd14_fp16_50": 82.84, "bk_fp16_50": 53.60,
          "bk_dpm20": 22.95, "bk_dpm20_taesd": 20.58, "bk_final": 20.58}

def load_pipeline(name, safety=False):
    cfg = CONFIGS[name]
    dtype = torch.float16 if (cfg["fp16"] and DEVICE != "cpu") else torch.float32
    kw = {} if safety else dict(safety_checker=None, requires_safety_checker=False)
    pipe = StableDiffusionPipeline.from_pretrained(cfg["model"], torch_dtype=dtype, **kw)
    if cfg["sched"] == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
    if cfg["taesd"]:
        pipe.vae = AutoencoderTiny.from_pretrained(TAESD, torch_dtype=dtype)
    if cfg["lora"]:
        if os.path.isdir(LORA):
            pipe.load_lora_weights(LORA)
            pipe.fuse_lora()          # merged into the UNet: zero extra inference cost
            pipe.p3_lora = True
        else:
            print(f"[warn] LoRA folder '{LORA}' not found - running WITHOUT LoRA")
            pipe.p3_lora = False
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    pipe.p3_name, pipe.p3_cfg = name, cfg
    return pipe

def generate(pipe, prompt, seed=0, res=512):
    """Returns (PIL image, seconds). CPU generator = identical starting noise on any device."""
    cfg = pipe.p3_cfg
    g = torch.Generator("cpu").manual_seed(int(seed))
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"],
               generator=g, height=res, width=res).images[0]
    if DEVICE == "cuda": torch.cuda.synchronize()
    return img, time.perf_counter() - t0

def model_size(pipe):
    """Parameters (millions) and in-memory MB per component actually used."""
    out = {}
    for part in ["text_encoder", "unet", "vae"]:
        m = getattr(pipe, part)
        n = sum(p.numel() for p in m.parameters())
        b = sum(p.numel() * p.element_size() for p in m.parameters())
        out[part] = dict(params_M=round(n / 1e6, 1), MB=round(b / 1e6, 1))
    out["total_MB"] = round(sum(v["MB"] for v in out.values() if isinstance(v, dict)), 1)
    return out
