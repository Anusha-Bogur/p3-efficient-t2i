# P3: Resource-Efficient Text-to-Image Generation (Team T38)

Live demo: <URL>  ·  Design document: docs/  ·  Results: eval_P3_T38.md

## What it does
BK-SDM-Small-2M (lightweight, knowledge-distilled Stable Diffusion) adapted with a LoRA on COCO,
sampled with DPM-Solver++ (20 steps) and decoded with TAESD, followed by an automatic classical-CV
pipeline (resize/upscale, colour correction, edge-aware filtering, quality assessment, artifact fixing).
Baseline: Stable Diffusion v1.4, 50 steps, FP32. Architecture: docs/architecture.png

## Quick start

Activate a Python virtual environment, then run these commands from the project folder.

```sh
python -m pip install -r requirements.txt
python app/app.py
```

Open the [local app](http://127.0.0.1:7860) after the model loads. The first startup downloads the BK-SDM and
TAESD weights. For CUDA on Windows, install a CUDA-enabled PyTorch build in the virtual environment.
At startup, the app downloads the team's LoRA from `anu-122006/p3-lora-bk` into `lora_bk/` or the
folder set by `P3_LORA`. The header shows `bk_final` when LoRA weights are available, and
`bk_dpm20_taesd` when the download fails and no local weights are available.

Enter a prompt or select an example, choose a generation mode, and choose whether to apply classical CV.
Low uses the standard fast pipeline. Compositional assigns regions to side-by-side subjects.
Medium selects the best of three candidates and downloads its CLIP ranking model on first use.
Use seed `-1` for a random seed, or reuse the seed shown in the run report.
Download the images from the comparison panel and expand **Run report** for the JSON.

For Hugging Face Spaces, copy `SPACE_README.md` to the Space repository as `README.md`, then
copy `requirements_space.txt` as `requirements.txt` and push the project files. Select ZeroGPU
or dedicated GPU hardware for the Space. ZeroGPU calls use the app's `spaces.GPU` wrapper.

## Reproduce results
    python eval/make_prompts.py --n 300      # needs data/coco/annotations/captions_val2017.json
    N=300 bash eval/run_all.sh               # 6 configurations -> results/summary.csv
## Train the LoRA
    python train/make_lora_data.py --n 1000  # needs captions_train2017.json
    bash train/lora.sh                       # -> lora_bk/

## Credits and licences
BK-SDM (Nota AI, CreativeML OpenRAIL-M) · Stable Diffusion v1.4 (CompVis, CreativeML OpenRAIL-M) ·
TAESD (madebyollin) · diffusers (Hugging Face) · MS COCO 2017 (annotations CC BY 4.0).
Model outputs are subject to the OpenRAIL-M use restrictions.
