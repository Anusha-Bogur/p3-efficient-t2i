# P3: Resource-Efficient Text-to-Image Generation for Mobile Devices: Design Document (as built)

Computer Vision Hackathon 2026 · Problem Statement 3 · Team T38
Live demo: https://huggingface.co/spaces/anu-122006/p3-efficient-t2i · Code: <GitHub URL>

> This document describes the system as implemented. It supersedes the earlier plan (P3_Design_Document.docx), which also covered LCM distillation and an INT8/OpenVINO tier; those were not shipped and are listed under future work.

## 1. Problem and goal

P3 asks for a text-to-image system that runs within mobile-class resource limits, reducing computation, memory and latency "without significantly compromising semantic alignment, visual quality and image diversity", and comparing quantitatively against a baseline model.

Our approach:
- Start from a **pretrained, knowledge-distilled lightweight diffusion model** (BK-SDM-Small-2M).
- **Adapt** it with a LoRA fine-tune.
- **Optimise the inference pipeline** with four strategies.
- Wrap it with an **automatic classical computer-vision pipeline** and **budget-aware generation modes**.
- Deploy it as a public web app and evaluate it against Stable Diffusion v1.4.

## 2. Architecture

See `docs/architecture.png` (Mermaid source: `docs/architecture.mmd`). One request flows through:
1. **App** (`app/app.py`): Gradio UI; input validation (empty prompt rejected, >77 CLIP tokens warned); random seed by default, reported for reproducibility.
2. **Mode dispatcher** (`models/modes.py`): Low, Compositional or Medium (section 6).
3. **Generation** (`models/pipeline.py`, config `bk_final`):
   - CLIP ViT-L/14 text encoder;
   - BK-SDM-Small-2M UNet with our LoRA fused in;
   - DPM-Solver++ for 20 steps (guidance 7.5);
   - TAESD decoder;
   - Stable Diffusion safety checker.
4. **Classical CV** (`classical/postprocess.py`): five automatic operations with a quality gate (section 7).
5. **Output:** raw and corrected image, plus a JSON report (seed, timings, model size, mode details, CV report).

**Fallback:** if the GPU runs out of memory, the request is regenerated in Low mode with 10 steps; shared settings are restored afterwards.

## 3. Model choice

| Role | Model | Why |
|---|---|---|
| Baseline | Stable Diffusion v1.4 (CompVis), DDIM 50 steps, FP32 | BK-SDM was distilled from v1.4, so the comparison isolates our optimisations |
| Lightweight model | BK-SDM-Small-2M (Nota AI) | Block-removed, knowledge-distilled SD UNet: 0.48B vs 0.86B parameters, published metrics, standard diffusers layout, CreativeML OpenRAIL-M |
| Text encoder | CLIP ViT-L/14 (frozen) | Shared with SD v1.x |
| Decoder | TAESD (madebyollin) | Tiny autoencoder: about 141 vs 2,514 GFLOPs per 512px decode |

**Related work (on-device text-to-image):**

| System | Size | Status |
|---|---|---|
| SnapFusion | — | 8 steps on iPhone; weights not released |
| MobileDiffusion | ~0.5B | Benchmarked on iPhone 15 Pro; weights not released |
| SnapGen++ (small) | 0.4B | GenEval 0.66; weights not released |
| SANA-0.6B | 0.6B | GenEval 0.64; open weights |
| DreamLite | 0.39B | GenEval 0.72; 1024px in ~3 s on iPhone 17 Pro; gated weights |
| BK-SDM (ours) | 0.48B UNet | SD 1.4-level quality, open |

The newer models match prompts better mainly through stronger text encoders (LLM/VLM) and very large-scale training, which can't be reproduced in a hackathon. BK-SDM was the strongest open, phone-proven option compatible with the standard tooling.

## 4. Meaningful adaptation: LoRA fine-tune

| Item | Value |
|---|---|
| Data | 1,000 COCO train2017 image–caption pairs (train split only: no overlap with evaluation prompts) |
| Method | LoRA on the UNet (diffusers `train_text_to_image_lora.py`), rank 16 |
| Settings | 1,000 steps, batch 4, learning rate 1e-4 (constant), fp16, 512px, centre crop + random flip, seed 0 |
| Hardware / time | Kaggle NVIDIA T4, 17 min 22 s; final step loss 0.141 |
| Weights | https://huggingface.co/anu-122006/p3-lora-bk (fused into the UNet at load time, so it adds no inference cost) |

**Effect:** in the visual comparison (`results/lora_before_after.png`), colours are more natural and less oversaturated, with composition unchanged. It's a light, style-level adaptation. The quantitative effect is row 5 vs row 6 of the ablation (eval file).

## 5. Inference optimisation (four strategies)

| # | Strategy | Change | Compute per 512px image |
|---|---|---|---|
| 1 | Lightweight architecture | SD v1.4 UNet → BK-SDM-Small UNet (803 → 511 GFLOPs per UNet pass) | 82.84 → 53.60 TFLOPs |
| 2 | Fewer sampling steps | DDIM 50 → DPM-Solver++ 20 | → 22.95 TFLOPs |
| 3 | Reduced precision | FP32 → fp16 | Same FLOPs; lower memory and latency |
| 4 | Lightweight decoder | SD VAE → TAESD | → 20.58 TFLOPs |

Overall: **about 4.0× less compute** than the baseline. The LoRA is fused, so it costs nothing at inference. Measured results are in the eval file.

**Mobile resource requirements:** the final model is about 1.22 GB in fp16 (text encoder + UNet + TAESD). Peak GPU memory is 1.41 GB on a T4 at 512px. Every shipped mode uses only forward passes (no gradients), so all of them map onto mobile runtimes (Core ML, QNN, TFLite). INT8 quantisation (UNet ~0.48 GB on disk, verified in preflight tests) is future work.

## 6. Prompt encoding and adaptive generation modes

Prompt encoding works on two paths:
- **Standard path (Low; all ablation rows):** CLIP encodes the whole prompt; the UNet is conditioned through cross-attention.
- **Compositional path** (`models/regional.py`), our own encoding logic:
  1. **Parse** the prompt into objects and a side-by-side relation ("beside", "next to", "left/right of", "and").
  2. **Encode** each object phrase separately, plus the full prompt and an empty prompt.
  3. **Condition** each object's encoding on its own soft canvas region (columns). At each step, region predictions are blended with the global prediction: global weight 0.15 for the first 40% of steps (layout), then 0.4 (coherence).
  4. "On/under" and counting prompts fall back to the standard path, because splitting the canvas hurt them in our visual test.

**Modes (advanced requirement 5: adaptive generation based on compute budget):**

| Mode | What it does | Time (T4) | Runs on a phone? |
|---|---|---|---|
| Low | Standard generation | ~1.9–2.2 s | Yes |
| Compositional | Regional prompting for side-by-side multi-object prompts | ~3.5 s | Yes (forward passes only) |
| Medium | 3 candidates (seeds s, s+1, s+2) at 25 steps, decoded with the full SD VAE; CLIP ViT-B/32 picks the best match | ~7.8 s | Yes (~4× time) |

**Fallback (advanced requirement 6):** a GPU out-of-memory error in any mode triggers a regeneration in Low mode with 10 steps (TAESD), reported in the run report as `fallback` and `effective_mode`.

## 7. Classical computer-vision pipeline

Every operation runs only when a measured statistic calls for it. Corrections are kept only if the quality score improves.

| # | Operation | Implementation | Trigger |
|---|---|---|---|
| 1 | Resizing / resolution | Lanczos upscale to 512px + unsharp mask applied only on dilated Canny edges (80/160) | Image not 512×512 |
| 2 | Colour correction | Gray-world white balance; CLAHE on L (clip 2.0, 8×8 tiles); saturation ×1.15 / ×0.88 | Cast > 25; L contrast < 40; colourfulness < 20 or > 110 |
| 3 | Edge-aware post-processing | Bilateral filter (d = 5, σ colour 30, σ space 5) with Canny edges protected | Noise estimate ≥ 3 |
| 4 | Quality assessment | Q = 0.35·sharpness + 0.25·exposure + 0.25·contrast + 0.15·(1 − noise) − clipped fraction (each term normalised) | Always |
| 5 | Artifact detection and correction | Black frame (reported); clipping > 2% → tone curve; blur (Laplacian variance < 40) → unsharp; FFT grid peak > 40× median → 3×3 median | Per detector |

**Gate:** if Q after correction < Q before, the uncorrected image is kept. A black frame (usually the safety filter) is reported, never "enhanced".

## 8. Evaluation protocol

- **Prompts:** 300 COCO val2017 captions (one per image, fixed seed 2026); generation seed = COCO image ID; 512×512; the same prompts and seeds for every configuration.
- **Metrics, in the PDF's order:**
  - CLIP score (ViT-L/14, 0–100);
  - FID (clean-fid against the same 300 real images, labelled **FID-300**: a small sample, not comparable with published FID-30K);
  - latency (median and p95 after 2 warm-ups);
  - peak GPU memory;
  - model size;
  - quality vs compute (TFLOPs per image).
- **Hardware:** all latency and memory rows on an NVIDIA T4 (Kaggle).
- **Ablation:** 6 rows, each changing one thing (eval file).
- **Compositional study:** 30 multi-object/colour prompts × 3 modes. Scores:
  - CLIP-L/14;
  - "all objects present" (OWLv2 open-vocabulary detection, score ≥ 0.2);
  - "objects + colours correct" (OWLv2 box crop → CLIP-B/32 colour classification).

## 9. Results summary

- **Final configuration `bk_final`** (row 6): CLIP-L/14 **26.04**, FID-300 **131.04**, latency **2.17 s** median (p95 2.21 s), peak memory **1.41 GB**, size **1,219 MB**, **20.58 TFLOPs** per image.
- **Baseline comparison (rows 1–5):** see `eval_P3_T38.md`.

**Compositional study (30 prompts, `bk_final`):**

| Mode | CLIP-L/14 | All objects present | Objects + colours correct | Latency |
|---|---|---|---|---|
| Low | 26.41 | 46.7% | 23.3% | 1.93 s |
| Compositional | 26.68 | 50.0% | 33.3% | 3.49 s |
| Medium | **27.75** | **56.7%** | **43.3%** | 7.79 s |

**Reading the results:**
- **Medium** (3 candidates, 25 steps, full VAE) gives the clearest gain: +1.3 on the independent ViT-L/14 score and +3 prompts with every object present. Its colour figure is optimistic, because CLIP-B/32 both selects its candidates and checks colours.
- **Compositional** improves side-by-side prompts modestly (+3 prompts with objects and colours correct). In `results/compositional/modes_comparison.jpg` it fixed pizza + juice, horse + cow and the yellow bench, but lost an object in three other prompts.
- With 30 prompts, one prompt equals 3.3 points.

## 10. Deployment

- Hugging Face Space on **ZeroGPU** (free shared GPU, permanent URL).
- Each visitor's request uses their own daily GPU allowance; one image uses about 2–8 s of it.
- The LoRA is downloaded from the Hugging Face Hub at start-up.
- A GitHub Actions workflow opens the Space every 6 hours, so it doesn't sleep before judging.
- Demo latencies (on ZeroGPU) differ from the reported T4 latencies; all reported numbers are from the T4.

## 11. Limitations

- **Semantic accuracy** is limited by the CLIP text encoder and the small UNet. In fast mode, about 40% of multi-object prompts were fully correct (manual review). The main failures are a missing second object, colour swaps, merged objects, and an attribute turned into the wrong object.
- **Quality score blind spot:** the classical-CV quality score doesn't measure colour balance. In one failure case, white balance tinted a red-dominated scene toward cyan while the score rose (0.948 → 0.976); see `results/failures/notes.md`.
- **Not solved by any mode:** actions and interactions ("dogs playing", "riding"), exact counts, overlapping objects ("on/under"), and text in images.
- **Small samples:** FID-300 and the 30-prompt compositional set. Differences of a few points are within noise.
- **LoRA:** a light adaptation (1,000 pairs, 17 minutes); its effect is small.
- **Server-only extras:** Attend-and-Excite and verified retries are implemented in `models/modes.py` but not exposed in the app and not evaluated on the full model, because they need gradients or extra detector models.

## 12. Ethics, safety and licences

- **Safety:** the Stable Diffusion safety checker is enabled in the app on both generation paths; flagged images are returned black and reported.
- **No personal data** is collected. The seed is the only stored per-request value, and it's shown only to the user.
- **Misuse:** generated images can be mistaken for real photos. Model outputs are subject to the OpenRAIL-M use restrictions.
- **Licences:**
  - BK-SDM, SD v1.4 and our LoRA: CreativeML OpenRAIL-M.
  - TAESD: MIT.
  - CLIP: MIT.
  - OWLv2: Apache 2.0.
  - spaCy: MIT.
  - diffusers: Apache 2.0.
  - COCO annotations: CC BY 4.0 (images under Flickr terms).
  - Our code: Apache 2.0.
- **Secrets:** tokens are stored only as platform secrets, never in the repository.

## 13. Future work

- INT8 (OpenVINO/QNN) export and real-phone profiling (Qualcomm AI Hub).
- LCM-LoRA distillation (4 steps).
- A model with a stronger text encoder (e.g. SANA-0.6B, DreamLite).
- A LoRA trained on attribute-rich compositional captions.
