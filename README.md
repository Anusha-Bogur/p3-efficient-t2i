# P3: Resource-Efficient Text-to-Image Generation (Team T__)

Live demo: <URL>  ·  Design document: docs/  ·  Results: eval_P3_TXX.md

## What it does
BK-SDM-Small-2M (lightweight, knowledge-distilled Stable Diffusion) adapted with a LoRA on COCO,
sampled with DPM-Solver++ (20 steps) and decoded with TAESD, followed by an automatic classical-CV
pipeline (resize/upscale, colour correction, edge-aware filtering, quality assessment, artifact fixing).
Baseline: Stable Diffusion v1.4, 50 steps, FP32. Architecture: docs/architecture.png

## Quick start
    pip install -r requirements.txt
    python app/app.py                        # local demo
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
