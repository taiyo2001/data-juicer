import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "sd_embed/src"))
# --------------------------------

import torch
from diffusers import DiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15
import json
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument('--model_name', type=str, default="SD1_5")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    parser.add_argument('--icl_num', type=str, default=None)
    parser.add_argument('--icl_prompt', type=str, default=None)

    args=parser.parse_args()

    args.output_json = f"./evaluation_pipeline/image_generation_example/output_image_info_{args.model_name}.json"
    args.image_output_dir = f"./evaluation_pipeline/image_generation_example/output_image_{args.model_name}/"

    return args


if __name__ == "__main__":
    args = parse_args()

    prompt_weighting = False
    if "_EM" in args.model_name:
        prompt_weighting = True

    print("--- Model Name: ", args.model_name, " ---")
    print("--- Prompt Weighting: ", prompt_weighting, " ---")
    print("--- ICL Num: ", args.icl_num, " ---")
    print("--- ICL Prompt: ", args.icl_prompt, " ---")

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    new_data = []

    if args.image_output_dir:
        os.makedirs(args.image_output_dir, exist_ok=True)

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            prompt = temp_piece["polished_prompt"]
            if args.icl_prompt and args.icl_num:
                prompt = args.icl_prompt + "\n" + prompt

            if prompt_weighting:
                (prompt_embeds, prompt_neg_embeds) = get_weighted_text_embeddings_sd15(
                    pipe, prompt=prompt, neg_prompt=""
                )
                image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=prompt_neg_embeds,
                    height=512,
                    width=512,
                    num_inference_steps=30,
                ).images[0]
            else:
                image = pipe(
                    prompt,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                ).images[0]

            image_name = f"{temp_piece['dataset_target']}_{args.model_name}_{valid_image_count}_{temp_piece['image_id']}"
            image.save(os.path.join(args.image_output_dir, image_name))

            temp_json = {}
            temp_json["output_image_name"] = image_name
            temp_json["image_id"] = temp_piece["dataset_target"] + "_" + temp_piece["image_id"]
            new_data.append(temp_json)

            valid_image_count += 1

        except Exception as e:
            print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
            print(e)
            print("----------------------------------------------------------")
        # except:
        #     continue

    with open(args.output_json, "a") as f:
        json.dump(new_data, f)
