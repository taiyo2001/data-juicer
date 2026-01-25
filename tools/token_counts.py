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

prompt_path = "./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json"
st1_prompt_path = "./data-juicer/outputs/llm_prompt/output_st_info_gemini-2.5-flash-lite.json"
st3_prompt_path = "./data-juicer/outputs/llm_prompt/output_st1_info_gemini-3-flash-preview.json"
dp_prompt_path = "./data-juicer/outputs/llm_prompt/output_dp1_info_gemini-3-flash-preview.json"
model_path = "black-forest-labs/FLUX.1-schnell"
model_qwen_path = "Qwen/Qwen-Image"
lora_qwen_path = "lightx2v/Qwen-Image-Lightning"
output_path = "./data-juicer/outputs/DetailMaster_Dataset_analysis.json"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)

pipe_qwen = DiffusionPipeline.from_pretrained(
    model_qwen_path,
    torch_dtype=torch.bfloat16,
)

token_stats = []

with open(prompt_path, "r") as f:
  data = json.load(f)

with open(st1_prompt_path, "r") as f:
  st1_data = json.load(f)

with open(st3_prompt_path, "r") as f:
  st3_data = json.load(f)

with open(dp_prompt_path, "r") as f:
  dp_data = json.load(f)

st1_dict = {item["image_id"]: item["llm_output"] for item in st1_data}
st3_dict = {item["image_id"]: item["llm_output"] for item in st3_data}
dp_dict = {item["image_id"]: item["llm_output"] for item in dp_data}

for temp_piece in tqdm.tqdm(data):
  try:
    image_id = f"{temp_piece['dataset_target']}_{temp_piece['image_id']}"

    prompt = temp_piece["polished_prompt"]
    st1_prompt = st1_dict.get(image_id)
    st3_prompt = st3_dict.get(image_id)
    dp_prompt = dp_dict.get(image_id)

    # --- Normal Prompt(CLIP) ---
    clip_inputs = pipe.tokenizer(
        prompt,
        padding="do_not_pad", # the same as False
        truncation=False,
        add_special_tokens=True
    )
    clip_token_count = len(clip_inputs["input_ids"])

    # --- Normal Prompt(T5) ---
    t5_inputs = pipe.tokenizer_2(
        prompt,
        padding="do_not_pad", # the same as False
        truncation=False,
        add_special_tokens=True
    )
    t5_token_count = len(t5_inputs["input_ids"])

    # --- Normal Prompt(Qwen2.5-VL) ---
    qwen_vl = pipe_qwen.tokenizer(
        prompt,
        padding="do_not_pad", # the same as False
        truncation=False,
        add_special_tokens=True
    )
    qwen_vl_token_count = len(qwen_vl["input_ids"])

    # --- ST1(Qwen2.5-VL) ---
    st1_qwen_vl_token_count = None
    if not st1_prompt is None:
      st1_qwen_vl = pipe_qwen.tokenizer(
          st1_prompt,
          padding="do_not_pad", # the same as False
          truncation=False,
          add_special_tokens=True
      )
      st1_qwen_vl_token_count = len(st1_qwen_vl["input_ids"])

    # --- ST3(Qwen2.5-VL) ---
    st3_qwen_vl_token_count = None
    if not st3_prompt is None:
      st3_qwen_vl = pipe_qwen.tokenizer(
          st3_prompt,
          padding="do_not_pad", # the same as False
          truncation=False,
          add_special_tokens=True
      )
      st3_qwen_vl_token_count = len(st3_qwen_vl["input_ids"])

    # --- DP(Qwen2.5-VL) ---
    dp_qwen_vl_token_count = None
    if not dp_prompt is None:
      dp_qwen_vl = pipe_qwen.tokenizer(
          dp_prompt,
          padding="do_not_pad", # the same as False
          truncation=False,
          add_special_tokens=True
      )
      dp_qwen_vl_token_count = len(dp_qwen_vl["input_ids"])

    token_stats.append({
        "image_id": temp_piece['image_id'],
        "clip_tokens": clip_token_count,
        "t5_tokens": t5_token_count,
        "qwen_vl_tokens": qwen_vl_token_count,
        "st1_qwen_vl_tokens": st1_qwen_vl_token_count,
        "st3_qwen_vl_tokens": st3_qwen_vl_token_count,
        "dp_qwen_vl_tokens": dp_qwen_vl_token_count,
        "prompt_preview": prompt[:50] + "..."
    })

  except Exception as e:
      print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
      print(e)
      print("----------------------------------------------------------")

with open(output_path, "w") as f:
    json.dump(token_stats, f, indent=4)
