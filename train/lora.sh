#!/bin/bash
# P1: LoRA adaptation of BK-SDM-Small-2M on 1,000 COCO train pairs (~1 h on a T4/L4-class GPU).
# Needs: git clone https://github.com/huggingface/diffusers (for the example script)
accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=nota-ai/bk-sdm-small-2m \
  --train_data_dir=data/coco_lora --caption_column=text \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 --gradient_accumulation_steps=1 \
  --max_train_steps=1000 --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 \
  --rank=16 --mixed_precision=fp16 --checkpointing_steps=250 \
  --dataloader_num_workers=2 --seed=0 --output_dir=lora_bk 2>&1 | tee results/lora_train.log
# Output: lora_bk/pytorch_lora_weights.safetensors  (loaded by models/pipeline.py, config bk_final)
