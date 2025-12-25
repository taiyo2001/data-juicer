import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

import torch
import json
import tqdm
from diffusers import DiffusionPipeline

prompt_path = "./DetailMaster_Dataset/DetailMaster_Dataset.json"
model_path = "black-forest-labs/FLUX.1-schnell"
output_path = "./outputs/DetailMaster_Dataset_analysis.json"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

token_stats = []

with open(prompt_path, "r") as f:
  data = json.load(f)

for temp_piece in tqdm.tqdm(data):
  try:
    prompt = temp_piece["polished_prompt"]

    # --- CLIP ---
    clip_inputs = pipe.tokenizer(
        prompt,
        padding="do_not_pad", # the same as False
        truncation=False,
        add_special_tokens=True
    )
    clip_token_count = len(clip_inputs["input_ids"])

    # --- T5 ---
    t5_inputs = pipe.tokenizer_2(
        prompt,
        padding="do_not_pad", # the same as False
        truncation=False,
        add_special_tokens=True
    )
    t5_token_count = len(t5_inputs["input_ids"])

    token_stats.append({
        "image_id": temp_piece['image_id'],
        "clip_tokens": clip_token_count,
        "t5_tokens": t5_token_count,
        "prompt_preview": prompt[:50] + "..."
    })

  except Exception as e:
      print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
      print(e)
      print("----------------------------------------------------------")

with open(output_path, "w") as f:
    json.dump(token_stats, f, indent=4)
