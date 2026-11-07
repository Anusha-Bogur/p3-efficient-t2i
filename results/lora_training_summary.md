# LoRA training summary (P1)
- Base model: nota-ai/bk-sdm-small-2m
- Data: 1,000 COCO train2017 image-caption pairs (train split only; no overlap with the val prompts)
- Script: diffusers examples/text_to_image/train_text_to_image_lora.py (train/lora.sh)
- Settings: 1,000 steps, batch 4, rank 16, learning rate 1e-4 (constant), fp16, resolution 512, center crop + random flip, seed 0
- Hardware: Kaggle NVIDIA Tesla T4 (1 GPU); time 17 min 22 s (~1.04-1.07 s/step)
- Final step loss: 0.141
- Weights: https://huggingface.co/anu-122006/p3-lora-bk (pytorch_lora_weights.safetensors), fused into the UNet at load time
- Problems fixed: Kaggle's torchao 0.10.0 clashed with PEFT (uninstalled); torchmetrics CLIPScore broke with transformers 5.x (eval/metrics.py rewritten)

## Progress lines captured in the notebook
```
Steps:  23%|██▎       | 226/1000 [03:44<13:49,  1.07s/it, lr=0.0001, step_loss=0.16]  still running
Steps:  36%|███▌      | 355/1000 [06:00<11:20,  1.06s/it, lr=0.0001, step_loss=0.0248]still running
Steps:  65%|██████▌   | 653/1000 [11:15<06:06,  1.06s/it, lr=0.0001, step_loss=0.163] still running
Steps: 100%|██████████| 1000/1000 [17:22<00:00,  1.07s/it, lr=0.0001, step_loss=0.141] There are modules in UNet2DConditionModel that should be kept in float32: []. Casting directly with `to()` can lead to inconsistent results; set `torch_dtype` in `from_pretrained()` instead to keep these modules in float32.
Steps: 100%|██████████| 1000/1000 [17:22<00:00,  1.04s/it, lr=0.0001, step_loss=0.141]
finished, exit code 0
```
