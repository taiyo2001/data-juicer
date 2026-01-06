import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "sd_embed/src"))
# GGUF_DEV_MODEL_PATH = os.path.join(PROJECT_ROOT, "data-juicer/models/flux1-dev-Q8_0.gguf")
GGUF_DEV_MODEL_PATH = os.path.join(PROJECT_ROOT, "data-juicer/models/flux1-dev-Q4_K_S.gguf")
# GGUF_SCHNELL_MODEL_PATH = os.path.join(PROJECT_ROOT, "data-juicer/models/flux1-schnell-Q8_0.gguf")
GGUF_SCHNELL_MODEL_PATH = os.path.join(PROJECT_ROOT, "data-juicer/models/flux1-schnell-Q4_K_S.gguf")

GGUF_MODEL_PATH = None
# --------------------------------

import torch
from diffusers import DiffusionPipeline, FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from sd_embed.src.sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
from src.constants import DETAIL_MASTER
import json
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="SD1_5")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    parser.add_argument('--count', type=str, default=None)
    # in-context learning
    parser.add_argument('--sp_num', type=str, default=None)
    parser.add_argument('--sp_prompt', type=str, default=None)
    parser.add_argument('--sp_t5_only', type=bool, default=False)
    # negative prompt
    parser.add_argument('--np_num', type=str, default=None)
    parser.add_argument('--np_prompt', type=str, default=None)

    args = parser.parse_args()

    if args.sp_num is not None and args.sp_prompt is not None:
        if args.sp_t5_only:
            args.model_name = args.model_name + f"_T5_SP{args.sp_num}"
        else:
            args.model_name = args.model_name + f"_SP{args.sp_num}"

    if args.np_num is not None and args.np_prompt is not None:
        args.model_name = args.model_name + f"_NP{args.np_num}"

    if args.count is not None:
        args.model_name = args.model_name + f"_{args.count}"

    args.output_json = f"./outputs/image_info/output_image_info_{args.model_name}.json"
    args.image_output_dir = f"./outputs/image/output_image_{args.model_name}/"

    return args


if __name__ == "__main__":
    args = parse_args()

    if 'dev' in args.model_name:
        args.model_path = "black-forest-labs/FLUX.1-dev"
        GGUF_MODEL_PATH = GGUF_DEV_MODEL_PATH
        num_inference_steps = 50
    elif 'schnell' in args.model_name:
        args.model_path = "black-forest-labs/FLUX.1-schnell"
        GGUF_MODEL_PATH = GGUF_SCHNELL_MODEL_PATH
        num_inference_steps = 4

    # NOTE: FLUX.1はdevのみprompt weighting対応（効果はイマイチらしい）
    prompt_weighting = False
    if "_EM" in args.model_name:
        if 'schnell' in args.model_name:
            print("!!! WARNING: FLUX.1-schnell does not support prompt weighting. Disabled prompt weighting. !!!")
            exit(1)

        prompt_weighting = True

    max_sequence_length = 256 # default
    if '_SL256' in args.model_name:
        max_sequence_length = 256
    elif '_SL512' in args.model_name:
        max_sequence_length = 512

    # --- Colab Configuration ---
    try:
        import google.colab
        is_colab = True
        DRIVE_PATH_BASE = '/content/drive/MyDrive/workspace/huggingface_cache/'
    except:
        is_colab = False
    print(f"--- is_colab: {is_colab} ---")
    # ----------------------------

    print("--- Model Name: ", args.model_name, " ---")
    print("--- Prompt Weighting: ", prompt_weighting, " ---")
    print("--- Max Sequence Length: ", max_sequence_length, " ---")
    print("--- SP Num: ", args.sp_num, " ---")
    print("--- SP Prompt: ", args.sp_prompt, " ---")
    print("--- Negative Prompt Num: ", args.np_num, " ---")
    print("--- Negative Prompt: ", args.np_prompt, " ---")

    if is_colab:
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    else:
        transformer = FluxTransformer2DModel.from_single_file(
            GGUF_MODEL_PATH,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        # GPU節約のためにto("cuda")の移行をなし(速度低下)
        pipe = FluxPipeline.from_pretrained(
            args.model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # ).to("cuda")


    new_data = []

    if args.image_output_dir:
        os.makedirs(args.image_output_dir, exist_ok=True)

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            # SD3.5の効果検証で精度が悪くなかったらFLUXにも適用
            # normal_prompt = temp_piece["polished_prompt"]
            # sp_positive_prompt = args.sp_prompt + "\n" + normal_prompt

            prompt = temp_piece["polished_prompt"]
            if args.sp_prompt and args.sp_num:
                prompt = args.sp_prompt + "\n" + prompt

            neg_prompt = ""
            if args.np_prompt and args.np_num:
                neg_prompt = args.np_prompt

            if prompt_weighting:
                print("!!! WARNING: Prompt weighting for FLUX.1 is not yet implemented. Generating without prompt weighting. !!!")
                # TODO: Search implementation for FLUX.1
                # prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                #     pipe  = pipe, prompt = prompt
                # )
                # image = pipe(
                #     prompt_embeds,
                #     pooled_prompt_embeds=pooled_prompt_embeds,
                #     height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                #     width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                #     num_inference_steps=num_inference_steps,
                # ).images[0]
            else:
                image = pipe(
                    prompt,
                    height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    num_inference_steps=num_inference_steps, # schnell: 4, dev: 50
                    max_sequence_length=max_sequence_length,
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
