"""P3 owns this file. Gradio app = the live URL (Hugging Face ZeroGPU Space). Also runs locally: python app/app.py"""
import sys, os, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import spaces                                     # Hugging Face ZeroGPU; must come before torch
    GPU = spaces.GPU(duration=45)
    os.environ.setdefault("P3_DEVICE", "cuda")        # must be set before models.pipeline is imported
except ImportError:                                   # running locally: no ZeroGPU wrapper needed
    GPU = lambda f: f
os.environ.setdefault("P3_LORA", os.path.join(ROOT, "lora_bk"))
try:                                                  # P1's LoRA, downloaded at start-up
    from huggingface_hub import snapshot_download
    snapshot_download("anu-122006/p3-lora-bk", local_dir=os.environ["P3_LORA"])
except Exception as e:
    print("[warn] LoRA download failed, running without LoRA:", e)
import torch, gradio as gr
from models.pipeline import load_pipeline, generate, model_size, CONFIGS, DEVICE, LORA
from models.modes import make_ae, generate_mode
from classical.postprocess import postprocess

NAME = "bk_final" if os.path.isfile(os.path.join(LORA, "pytorch_lora_weights.safetensors")) else "bk_dpm20_taesd"
PIPE = load_pipeline(NAME, safety=True)
SIZE = model_size(PIPE)
AE = make_ae(PIPE)
MODES = ["Low", "Compositional", "Medium"]

@GPU
def run(prompt, seed, use_cv, mode):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
    n_tok = len(PIPE.tokenizer(prompt).input_ids)
    note = "Prompt longer than 77 tokens: the end was truncated. " if n_tok > 77 else ""
    fallback = None
    try:
        img, t, extra = generate_mode(PIPE, AE, prompt, seed, mode)
    except torch.cuda.OutOfMemoryError:                       # resource-aware fallback
        torch.cuda.empty_cache()
        img, t, extra = generate_mode(PIPE, AE, prompt, seed, "Low")   # resets to Low settings first
        fallback = f"Out of GPU memory in {mode} mode: fell back to Low"
    out, report = postprocess(img) if use_cv else (img, {"classical_cv": "off"})
    info = dict(model=NAME, mode=mode, seed=seed, steps=CONFIGS[NAME]["steps"], device=DEVICE,
                generation_s=round(t, 2), model_size_MB=SIZE["total_MB"], fallback=fallback,
                note=note or None, mode_details=extra, classical_cv=report)
    return img, out, info

with gr.Blocks(title="P3: Resource-efficient text-to-image") as demo:
    gr.Markdown("## P3: Resource-efficient text-to-image\n"
                "BK-SDM-Small + DPM-Solver++ (20 steps) + TAESD + LoRA, followed by an automatic classical-CV pipeline. "
                f"Model: **{NAME}**.  \n**Modes:** Low = fastest (~2 s on a T4) · Compositional = side-by-side "
                "multi-object prompts: each object gets its own text encoding and canvas region (~3.5 s) · "
                "Medium = best of 3 candidates, picked by CLIP (~8 s).")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="a red car beside a blue house", scale=4)
        seed = gr.Number(label="Seed (-1 = random; reuse a seed to reproduce an image)", value=-1, precision=0, scale=1)
    with gr.Row():
        mode = gr.Radio(MODES, value="Low", label="Mode")
        use_cv = gr.Checkbox(label="Apply classical CV post-processing", value=True)
    btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        raw = gr.Image(label="Generated (raw)", type="pil")
        final = gr.Image(label="After classical CV", type="pil")
    info = gr.JSON(label="Timings, model size, seed, mode details and CV report")
    gr.Examples([["a red car beside a blue house", 1000, "Medium"],          # seeds = those used in the evaluation
                 ["a pizza next to a glass of orange juice", 1006, "Compositional"],
                 ["a black horse and a white cow in a field", 1004, "Medium"],
                 ["a golden retriever playing in a park", 0, "Low"]], [prompt, seed, mode])
    btn.click(run, [prompt, seed, use_cv, mode], [raw, final, info])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
