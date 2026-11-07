# P3: Resource-Efficient Text-to-Image Generation for Mobile Devices (Team T38)

**Live demo:** https://huggingface.co/spaces/anu-122006/p3-efficient-t2i
**Design document:** `docs/design_document.md` · **Architecture:** `docs/architecture.png` · **Results:** `eval_P3_T38.md`

## What it does
- **Lightweight model:** BK-SDM-Small-2M, a knowledge-distilled Stable Diffusion with 0.48B vs 0.86B UNet parameters.
- **Adaptation:** a LoRA fine-tuned on COCO.
- **Fast pipeline:** DPM-Solver++ (20 steps), fp16 and the TAESD decoder. About **4× less compute** than the Stable Diffusion v1.4 baseline, ~2 s per image on an NVIDIA T4.
- **Automatic classical-CV pipeline:** resize/upscale, colour correction, edge-aware filtering, quality assessment, artifact detection and correction.

## How to use the live demo
1. **Open the link** above, on a laptop or a phone. If the app has been idle, the first load can take a minute or two while the Space wakes up and loads the model.
2. **Type a prompt** under *Create an image*, or click one of the **example buttons** (Red car, Pizza and juice, Horse and cow, Golden retriever). The examples fill in a prompt, seed and mode from our evaluation.
3. **Choose a generation mode:**

   | Mode | Best for | Approx. time |
   |---|---|---|
   | **Low** | Fastest; single subjects and simple scenes | ~2 s |
   | **Compositional** | Two objects side by side, e.g. `a pizza next to a glass of orange juice`. Each object gets its own text encoding and canvas region. Prompts with "on/under" or counts automatically use Low | ~3.5 s |
   | **Medium** | Best prompt matching: generates 3 candidates and CLIP picks the one closest to the prompt | ~8 s |

   Times are measured on a T4; the demo GPU may be faster.

4. **Seed:** leave it at **-1** for a new random image each time. The seed actually used is shown in the summary; type it back in to reproduce that exact image.
5. **Classical CV:** keep *Apply classical CV post-processing* ticked to compare the **Raw image** with the **CV result**. If the quality check rejects the corrections, the CV result shows the raw image and the summary says so.
6. Click **Generate image**. The summary shows generation time, steps, model size and seed. **Run report (JSON)** lists the mode details (e.g. the regions used in Compositional mode), any fallback, and which classical-CV operations ran, with the quality score before and after.

### GPU usage (ZeroGPU)
The demo runs on a Hugging Face **ZeroGPU** Space: a free, shared GPU assigned only while an image is being generated.
- **Each visitor uses their own daily GPU allowance**, set by Hugging Face. At the time of writing: about 2 minutes per day without logging in, more for logged-in and PRO accounts.
- **One image uses only its generation time:** ~2 s (Low), ~3.5 s (Compositional), ~8 s (Medium). That's dozens of images per day even without logging in.
- **If the allowance runs out,** the app shows a quota message. Log in to Hugging Face for a larger allowance, or try again later.
- **Busy moments:** requests may briefly wait in a queue for a free GPU.
- **Timings shown in the demo can differ** from our reported numbers, which were all measured on an NVIDIA T4 (see `eval_P3_T38.md`).

## Run locally
    pip install -r requirements.txt
    python app/app.py            # downloads BK-SDM, TAESD and our LoRA automatically; needs a CUDA GPU for normal speed

## Reproduce the results
    python eval/make_prompts.py --n 300          # needs data/coco/annotations/captions_val2017.json
    N=300 bash eval/run_all.sh                   # ablation rows -> results/*/metrics.json, summary.csv
    python eval/compositional_eval.py --config bk_final --modes Low,Compositional,Medium
    python eval/plot_results.py                  # quality-vs-cost plot

## Train the LoRA
    python train/make_lora_data.py --n 1000      # 1,000 COCO train pairs (needs captions_train2017.json)
    bash train/lora.sh                           # -> lora_bk/  (published: huggingface.co/anu-122006/p3-lora-bk)

## Repository map
| Path | Contents |
|---|---|
| `app/app.py`, `app/lab.css` | Gradio app and its styling (ZeroGPU on Hugging Face; also runs locally) |
| `models/pipeline.py` | The 6 evaluated configurations, model loading, generation |
| `models/modes.py`, `models/regional.py` | Low / Compositional / Medium modes; regional prompting |
| `classical/postprocess.py` | The 5 classical-CV operations + quality gate |
| `eval/` | CLIP / FID / latency / memory / size evaluation, compositional evaluation, plot |
| `train/` | LoRA data preparation and training |
| `results/` | Metrics, comparison images, failure cases, training summary |
| `docs/` | Design document, architecture diagram (png + Mermaid), LoRA model card |
| `test_cv.py` | Synthetic tests that trigger each classical-CV operation |

## Credits and licences
- **Models:**
  - BK-SDM (Nota AI) and Stable Diffusion v1.4 (CompVis): CreativeML OpenRAIL-M.
  - Our LoRA: CreativeML OpenRAIL-M (derived from BK-SDM).
  - TAESD (madebyollin): MIT.
  - CLIP (OpenAI): MIT.
  - OWLv2 (Google): Apache 2.0.
- **Libraries:** diffusers / transformers (Hugging Face): Apache 2.0; spaCy: MIT.
- **Data:** MS COCO 2017 annotations: CC BY 4.0 (images under Flickr terms).
- **Our code:** Apache 2.0.

Model outputs are subject to the OpenRAIL-M use restrictions.
