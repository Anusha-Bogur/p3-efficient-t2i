---
license: creativeml-openrail-m
base_model: nota-ai/bk-sdm-small-2m
tags: [text-to-image, lora, diffusers, stable-diffusion, bk-sdm]
---
# p3-lora-bk: LoRA for BK-SDM-Small-2M (CV Hackathon 2026, P3)

(Paste this as README.md in the Hugging Face repo anu-122006/p3-lora-bk.)

- **Base model:** nota-ai/bk-sdm-small-2m
- **Data:** 1,000 COCO train2017 image–caption pairs
- **Training:** diffusers `train_text_to_image_lora.py`; UNet LoRA rank 16; 1,000 steps; batch 4; lr 1e-4; fp16; 512px; NVIDIA T4, 17 min
- **Use:** `pipe.load_lora_weights("anu-122006/p3-lora-bk"); pipe.fuse_lora()` on a BK-SDM-Small-2M pipeline
- **Effect:** light style adaptation toward COCO-like everyday photos; see the project's evaluation for measured results
- **Licence:** CreativeML OpenRAIL-M, inherited from the base model, with the same use restrictions
