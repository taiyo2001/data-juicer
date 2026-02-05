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
import pysbd
from diffusers import DiffusionPipeline

prompt_path = "./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json"
output_path = "./data-juicer/outputs/DetailMaster_Dataset_sentence_analysis.json"
model_path = "black-forest-labs/FLUX.1-schnell"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)

seg = pysbd.Segmenter(language='en', clean=False)

token_stats = []

with open(prompt_path, "r") as f:
    data = json.load(f)

for temp_piece in tqdm.tqdm(data):
    try:
        prompt = temp_piece["polished_prompt"]
        sentences = seg.segment(prompt)

        sentence_details = []
        all_clip_tokens = []

        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            # 文単位のCLIPトークン数計測 (BOS/EOSを含めない純粋な文の長さ)
            clip_inputs = pipe.tokenizer(
                sent,
                padding="do_not_pad", # the same as False
                truncation=False,
                add_special_tokens=False
            )
            clip_cnt = len(clip_inputs["input_ids"])
            all_clip_tokens.append(clip_cnt)

            sentence_details.append({
                "sentence_idx": idx,
                "text": sent,
                "clip_token_count": clip_cnt,
            })

        token_stats.append({
            "image_id": temp_piece['image_id'],
            "total_sentences": len(sentence_details),
            "total_clip_tokens": sum(all_clip_tokens),
            "avg_clip_tokens_per_sent": sum(all_clip_tokens) / len(all_clip_tokens) if all_clip_tokens else 0,
            "max_clip_tokens_in_sent": max(all_clip_tokens) if all_clip_tokens else 0,
            "sentences": sentence_details
        })

    except Exception as e:
        print(f"\n--- ERROR encountered for prompt {temp_piece.get('image_id')} ---")
        print(e)
        print("----------------------------------------------------------")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(token_stats, f, indent=4)
