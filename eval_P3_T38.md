USN1,USN2,USN3,USN4
# eval_P3_T38: Resource-efficient text-to-image (BK-SDM-Small-2M + LoRA + DPM-Solver++ + TAESD)

**Protocol:** 300 COCO val2017 captions (one per image, sampled with seed 2026); generation seed = COCO image ID; 512×512; identical prompts and seeds for every row. GPU: NVIDIA Tesla T4 (Kaggle), latency = median of 300 images after 2 warm-ups. Baseline: Stable Diffusion v1.4, DDIM 50 steps, FP32.

| # | Configuration | Changes vs previous row |
|---|---|---|
| 1 | sd14_fp32_50 | Baseline |
| 2 | sd14_fp16_50 | Reduced precision (fp16) |
| 3 | bk_fp16_50 | Lightweight architecture (BK-SDM-Small-2M) |
| 4 | bk_dpm20 | Fewer steps (DPM-Solver++ 20) |
| 5 | bk_dpm20_taesd | Lightweight decoder (TAESD) |
| 6 | **bk_final** | Meaningful adaptation (LoRA, fused): **final model** |

## 1. CLIP similarity (text–image alignment, ViT-L/14, 0–100, higher is better)
| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| [P2] | [P2] | [P2] | [P2] | [P2] | **26.04** |

## 2. FID-300 (clean-fid vs the same 300 real COCO images; lower is better)
Small-sample FID: only comparable between these rows, never with published FID-30K values.
| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| [P2] | [P2] | [P2] | [P2] | [P2] | **131.04** |

## 3. Generation latency (seconds per image, T4; median / p95)
| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| [P2] | [P2] | [P2] | [P2] | [P2] | **2.17 / 2.21** |

Laptop proxy (not a phone): [P2: device name, latency for bk_final, n = 5]

## 4. Peak memory (GPU, GB)
| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| [P2] | [P2] | [P2] | [P2] | [P2] | **1.41** |

## 5. Model size (MB; text encoder + UNet + decoder, in the precision used)
| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| [P2] | [P2] | [P2] | [P2] | [P2] | **1,219** |

## 6. Quality vs computational cost
| # | TFLOPs per image | CLIP-L/14 | Latency (s) |
|---|---|---|---|
| 1 | 82.84 | [P2] | [P2] |
| 2 | 82.84 | [P2] | [P2] |
| 3 | 53.60 | [P2] | [P2] |
| 4 | 22.95 | [P2] | [P2] |
| 5 | 20.58 | [P2] | [P2] |
| 6 | 20.58 | 26.04 | 2.17 |

Plot: `results/quality_vs_cost.png`. Final vs baseline: **4.0× less compute**; [P2: × faster, CLIP change].

## Ablation: what each step contributed
[P4: one line per row from the tables above, e.g. "Row 3 vs 2: BK-SDM cut latency by X% for a CLIP change of Y."]

## Compositional prompts (advanced requirements 1–2, 5)
30 multi-object/colour prompts (`eval/prompts_comp.py`), `bk_final`, seed = 1000 + prompt index, T4.

| Mode | CLIP-L/14 | All objects present | Objects + colours correct | Latency |
|---|---|---|---|---|
| Low | 26.41 | 46.7% | 23.3% | 1.93 s |
| Compositional | 26.68 | 50.0% | 33.3% | 3.49 s |
| Medium | 27.75 | 56.7% | 43.3% | 7.79 s |

- Object presence: OWLv2 (score ≥ 0.2).
- Colour: OWLv2 crop → CLIP-B/32 colour classification.
- Medium = 3 candidates at 25 steps decoded with the full SD VAE; CLIP-B/32 picks one. Its colour figure is optimistic, because it also selects with CLIP-B/32.
- One prompt = 3.3 points.

## Failure cases
See `results/failures/notes.md`.
- **Dominant failure in fast mode:** the second object is missing (8 of 30 multi-object prompts).
- **Then:** colour swaps (4), merged objects (2), an attribute turned into the wrong object (1).
- **Unsolved by any mode:** actions, exact counts, overlapping objects, text in images.

## Limitations
- Small-sample FID-300 and 30-prompt compositional set.
- Demo runs on ZeroGPU, while all reported numbers come from the T4.
- The LoRA is a light adaptation.
- Not measured on a phone; laptop proxy only.
