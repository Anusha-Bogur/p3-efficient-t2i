"""P3 owns this file. Gradio app = the live URL."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spaces                              # must come before torch
os.environ.setdefault("P3_DEVICE", "cuda")  # must be set before models.pipeline is imported
import torch, gradio as gr
from models.pipeline import load_pipeline, generate, model_size, CONFIGS, DEVICE, LORA
from classical.postprocess import postprocess

NAME = "bk_final" if os.path.isdir(LORA) else "bk_dpm20_taesd"
PIPE = load_pipeline(NAME, safety=True)
SIZE = model_size(PIPE)

@spaces.GPU(duration=45)

def run(prompt, seed, use_cv):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    n_tok = len(PIPE.tokenizer(prompt).input_ids)
    note = "Prompt longer than 77 tokens: the end was truncated. " if n_tok > 77 else ""
    steps = CONFIGS[NAME]["steps"]; fallback = None
    try:
        img, t = generate(PIPE, prompt, seed=int(seed))
    except torch.cuda.OutOfMemoryError:                       # resource-aware fallback
        torch.cuda.empty_cache(); PIPE.p3_cfg = dict(PIPE.p3_cfg, steps=10)
        img, t = generate(PIPE, prompt, seed=int(seed)); PIPE.p3_cfg = dict(PIPE.p3_cfg, steps=steps)
        fallback = "Out of GPU memory: retried with 10 steps"
    out, report = postprocess(img) if use_cv else (img, {"classical_cv": "off"})
    info = dict(model=NAME, steps=steps, device=DEVICE, generation_s=round(t, 2),
                model_size_MB=SIZE["total_MB"], fallback=fallback, note=note or None, classical_cv=report)
    return img, out, info

with gr.Blocks(title="P3: Resource-efficient text-to-image") as demo:
    gr.Markdown("## P3: Resource-efficient text-to-image\nBK-SDM-Small + DPM-Solver++ (20 steps) + TAESD + LoRA, "
                "followed by an automatic classical-CV pipeline. Model: " + NAME)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="a red car beside a blue house", scale=4)
        seed = gr.Number(label="Seed", value=0, precision=0, scale=1)
    use_cv = gr.Checkbox(label="Apply classical CV post-processing", value=True)
    btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        raw = gr.Image(label="Generated (raw)", type="pil")
        final = gr.Image(label="After classical CV", type="pil")
    info = gr.JSON(label="Timings, model size and CV report")
    gr.Examples([["a red car parked beside a blue house", 0], ["two dogs playing on green grass", 1],
                 ["a cup to the left of a plate on a wooden table", 2]], [prompt, seed])
    btn.click(run, [prompt, seed, use_cv], [raw, final, info])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
