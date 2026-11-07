"""Browser demo for the P3 text-to-image pipeline."""

import os
from pathlib import Path
import random
import sys
import time

try:
    import spaces  # ZeroGPU must be initialized before any torch imports.
except ImportError:
    spaces = None

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
STYLES = Path(__file__).with_name("lab.css")
MAX_SEED = 2**53 - 1
MODES = ("Low", "Compositional", "Medium")
EXAMPLES = (
    ("Red car", "a red car beside a blue house", 1000, "Medium"),
    ("Pizza and juice", "a pizza next to a glass of orange juice", 1006, "Compositional"),
    ("Horse and cow", "a black horse and a white cow in a field", 1004, "Medium"),
    ("Golden retriever", "a golden retriever playing in a park", 0, "Low"),
)
sys.path.insert(0, str(ROOT))


def make_runner(name, pipe, ae):
    import torch

    from classical.postprocess import postprocess
    from models.modes import generate_mode
    from models.pipeline import DEVICE, generate, model_size

    def run(prompt, seed, use_cv, mode="Low"):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Enter a prompt to generate an image.", print_exception=False)
        if seed is not None and (
            not isinstance(seed, (int, float))
            or not -MAX_SEED <= seed <= MAX_SEED
            or int(seed) != seed
        ):
            raise gr.Error("Enter a whole-number seed within the supported range.", print_exception=False)
        seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else int(seed)
        if mode not in MODES:
            raise gr.Error("Choose Low, Compositional, or Medium mode.", print_exception=False)

        token_limit = pipe.tokenizer.model_max_length
        token_count = len(pipe.tokenizer(prompt, truncation=False, verbose=False).input_ids)
        note = None
        if token_count > token_limit:
            note = f"The prompt exceeded {token_limit} tokens. Its ending was truncated."

        config = pipe.p3_cfg
        decoder, ae_decoder = pipe.vae, ae.vae
        fallback = None
        try:
            try:
                raw, elapsed, mode_details = generate_mode(pipe, ae, prompt, seed, mode)
            except torch.cuda.OutOfMemoryError:
                fallback = f"The GPU ran out of memory in {mode} mode. Retried in Low mode with 10 steps."

            if fallback:
                torch.cuda.empty_cache()
                pipe.vae = ae.vae = pipe.p3_taesd
                pipe.p3_cfg = {**config, "steps": 10}
                try:
                    raw, elapsed = generate(pipe, prompt, seed=seed)
                except torch.cuda.OutOfMemoryError as exc:
                    raise gr.Error("The GPU ran out of memory. Try again shortly.") from exc
                mode_details = {}

            steps = pipe.p3_cfg["steps"]
            size_mb = model_size(pipe)["total_MB"]
        finally:
            # The pipeline and both decoders are shared by queued requests.
            pipe.p3_cfg = config
            pipe.vae, ae.vae = decoder, ae_decoder

        output, report, cv_seconds = raw, {"classical_cv": "off"}, 0.0
        if use_cv:
            start = time.perf_counter()
            output, report = postprocess(raw)
            cv_seconds = time.perf_counter() - start

        return raw, output, {
            "model": name,
            "mode": mode,
            "effective_mode": "Low" if fallback else mode,
            "mode_details": mode_details,
            "seed": seed,
            "steps": steps,
            "device": DEVICE,
            "generation_s": round(elapsed, 2),
            "cv_s": round(cv_seconds, 3),
            "model_size_MB": size_mb,
            "fallback": fallback,
            "note": note,
            "classical_cv": report,
        }

    return run


def describe_run(report):
    summary = [
        f"**{report['generation_s']:.2f} s** generation",
        f"**{report['steps']}** steps",
        f"**{report['model_size_MB']:.1f} MB** model",
        f"**Seed** {report['seed']}",
    ]
    cv = report["classical_cv"]
    cv_label = "CV result"
    notes = [report["note"], report["fallback"]]
    if cv.get("classical_cv") == "off":
        cv_label = "CV off"
        notes.append("Both previews show the raw image.")
    elif "black_frame" in cv.get("artifacts_found", []):
        notes.append("The model returned a black image. Try a different prompt.")
    elif not cv["correction_kept"]:
        notes.append("The CV quality check kept the raw image.")

    body = "\n\n".join([" · ".join(summary), *(note for note in notes if note)])
    return gr.update(value=body, visible=True), gr.update(label=cv_label), gr.update(visible=True)


def build_demo(model_name, run):
    with gr.Blocks(title="P3 Image Lab", fill_width=True) as demo:
        with gr.Column(elem_id="image-lab"):
            gr.HTML(
                f"""
                <header class="lab-hero">
                    <div class="lab-masthead">
                        <span class="lab-brand">P3 Image Lab</span>
                        <span class="lab-model" aria-label="Active model: {model_name}">{model_name}</span>
                    </div>
                    <div class="lab-intro">
                        <p class="lab-kicker">Computer Vision Hackathon</p>
                        <h1>Turn a prompt into an image</h1>
                        <p>Generate with BK-SDM Small and see what classical computer vision changes.</p>
                    </div>
                </header>
                """,
                apply_default_css=False,
                elem_classes="lab-static",
            )
            with gr.Row(elem_id="lab-workspace"):
                with gr.Column(min_width=0, elem_classes=["lab-panel", "lab-controls"]):
                    gr.Markdown("## Create an image", elem_classes="lab-panel-heading")
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A red car parked beside a blue house",
                        lines=4,
                        elem_classes="lab-input",
                    )
                    gr.Markdown("Try an example", elem_classes="lab-example-label")
                    with gr.Row(elem_id="lab-examples"):
                        examples = [
                            gr.Button(label, size="sm", min_width=0) for label, _, _, _ in EXAMPLES
                        ]
                    mode = gr.Radio(
                        choices=list(MODES), value="Low", label="Generation mode",
                        info="Low for speed, Compositional for side-by-side subjects, Medium to select the best of three.",
                        elem_classes="lab-input",
                    )
                    seed = gr.Number(
                        label="Seed", value=-1, info="Use -1 for a random seed, or reuse a reported seed.",
                        elem_classes="lab-input",
                    )
                    use_cv = gr.Checkbox(label="Apply classical CV post-processing", value=True, container=False)
                    generate_button = gr.Button("Generate image", variant="primary", elem_id="lab-generate")

                with gr.Column(min_width=0, elem_classes="lab-panel"):
                    gr.HTML(
                        """
                        <div class="lab-results-heading">
                            <h2>Image comparison</h2>
                            <span>512 × 512</span>
                        </div>
                        """,
                        apply_default_css=False,
                        elem_classes="lab-static",
                    )
                    with gr.Row(elem_id="lab-images"):
                        raw = gr.Image(
                            label="Raw image", type="pil", format="png", min_width=0,
                            interactive=False, buttons=["download", "fullscreen"],
                            elem_classes="lab-image",
                        )
                        output = gr.Image(
                            label="CV result", type="pil", format="png", min_width=0,
                            interactive=False, buttons=["download", "fullscreen"],
                            elem_classes="lab-image",
                        )
                    summary = gr.Markdown(visible=False, elem_id="lab-run-summary")
                    with gr.Accordion("Run report (JSON)", open=False, visible=False) as report_panel:
                        report = gr.JSON(show_label=False)

        for example, (_, text, example_seed, example_mode) in zip(examples, EXAMPLES):
            example.click(
                lambda text=text, seed=example_seed, mode=example_mode: (text, seed, mode),
                outputs=[prompt, seed, mode], queue=False, api_visibility="private",
            )
        generation = generate_button.click(
            run, [prompt, seed, use_cv, mode], [raw, output, report],
            api_name="run", concurrency_limit=1, show_progress_on=[generate_button, raw, output],
        )
        generation.success(
            describe_run, report, [summary, output, report_panel],
            queue=False, api_visibility="private", show_progress="hidden",
        )

    return demo.queue(max_size=20, api_open=False)


def lab_theme():
    return gr.themes.Soft(
        primary_hue="teal", neutral_hue="slate", font=["Segoe UI", "system-ui", "sans-serif"]
    ).set(
        background_fill_primary="#ffffff",
        background_fill_secondary="#f1f5f5",
        body_text_color_subdued="#526473",
        block_info_text_color="#526473",
        block_title_background_fill="transparent",
        block_title_text_color="*body_text_color",
        block_label_background_fill="*background_fill_primary",
        block_label_text_color="*body_text_color",
        input_background_fill="#f7f9fa",
        input_placeholder_color="#526473",
        input_shadow="none",
        button_primary_background_fill="#14695f",
        button_primary_background_fill_hover="#10564e",
        button_primary_text_color="#ffffff",
        button_primary_shadow="none",
        button_secondary_border_color="#dbe4e4",
        button_secondary_shadow="none",
    )


def main():
    if spaces is not None and os.environ.get("SPACE_ID"):
        os.environ.setdefault("P3_DEVICE", "cuda")
    os.environ.setdefault("P3_LORA", str(ROOT / "lora_bk"))

    try:
        from huggingface_hub import snapshot_download

        snapshot_download("anu-122006/p3-lora-bk", local_dir=os.environ["P3_LORA"])
    except Exception as exc:
        print(f"[warn] LoRA download failed, using locally available weights: {exc}")

    from models.modes import make_ae
    from models.pipeline import LORA, load_pipeline

    weights = Path(LORA)
    has_lora = any(
        path.is_file() for pattern in ("*.safetensors", "*.bin") for path in weights.glob(pattern)
    )
    name = "bk_final" if has_lora else "bk_dpm20_taesd"
    pipe = load_pipeline(name, safety=True)
    ae = make_ae(pipe)
    run = make_runner(name, pipe, ae)
    if spaces is not None:
        run = spaces.GPU(duration=45)(run)
    build_demo(name, run).launch(
        theme=lab_theme(),
        css_paths=STYLES,
        footer_links=["api"],
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
