import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

import torch
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel
from sd_embed.src.sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from src.services.slack_client_service import slack_service, build_image_generation_start_message, build_image_generation_complete_message
from src.constants import DETAIL_MASTER
import json
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="stabilityai/stable-diffusion-3.5-large-turbo")
    parser.add_argument('--model_name', type=str, default="SD3_5_large_turbo")
    parser.add_argument('--prompt_path', type=str, default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
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
    parser.add_argument('--np_prompt_path', type=str, default=None)

    args=parser.parse_args()

    if args.sp_num is not None and args.sp_prompt is not None:
        if args.sp_t5_only:
            args.model_name = args.model_name + f"_T5_SP{args.sp_num}"
        else:
            args.model_name = args.model_name + f"_SP{args.sp_num}"

    if args.np_num is not None and (args.np_prompt is not None or args.np_prompt_path is not None):
        args.model_name = args.model_name + f"_NP{args.np_num}"

    if args.count is not None:
        args.model_name = args.model_name + f"_{args.count}"

    args.output_json = f"./data-juicer/outputs/image_info/output_image_info_{args.model_name}.json"
    args.image_output_dir = f"./data-juicer/outputs/image/output_image_{args.model_name}/"

    return args


if __name__ == "__main__":
    args = parse_args()

    if 'large-turbo' in args.model_name:
        args.model_path = "stabilityai/stable-diffusion-3.5-large-turbo"
        num_inference_steps = 4
        guidance_scale = 0.0 # No CFG
    elif 'large' in args.model_name:
        args.model_path = "stabilityai/stable-diffusion-3.5-large"
        num_inference_steps = 28
        guidance_scale = 4.0
    elif 'medium' in args.model_name:
        args.model_path = "stabilityai/stable-diffusion-3.5-medium"
        num_inference_steps = 40
        guidance_scale = 7.0 # or 4.0(sd_embed)

    prompt_weighting = False
    extend_clip = False
    exted_t5 = False
    if "_EM" in args.model_name:
        prompt_weighting = True
        if '_EM-C' in args.model_name:
            extend_clip = True
        elif '_EM-T' in args.model_name:
            exted_t5 = True
        elif '_EM-A' in args.model_name:
            extend_clip = True
            exted_t5 = True

    max_sequence_length = 256 # default
    if '_SL256' in args.model_name:
        max_sequence_length = 256
    elif '_SL512' in args.model_name:
        max_sequence_length = 512

    # --- Colab Configuration ---
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
    print(f"--- is_colab: {is_colab} ---")
    # ----------------------------

    print("--- Model Name: ", args.model_name, " ---")
    print("--- Prompt Weighting: ", prompt_weighting, ", CLIP: ", extend_clip, ", T5: ", exted_t5, " ---")
    print("--- Max Sequence Length: ", max_sequence_length, " ---")
    print("--- SP Num: ", args.sp_num, " ---")
    print("--- SP Prompt: ", args.sp_prompt, " ---")
    print("--- Negative Prompt Num: ", args.np_num, " ---")
    print("--- Negative Prompt: ", args.np_prompt_path or args.np_prompt, " ---")

    # notify start
    mention_id = os.environ.get("SLACK_MENTION_ID")
    model_info = {
        "Prompt Weighting": f"{prompt_weighting}, CLIP: {extend_clip}, T5: {exted_t5}",
        "Max Sequence Length": max_sequence_length,
        "SP Num": args.sp_num,
        "SP Prompt": args.sp_prompt,
        "Negative Prompt Num": args.np_num,
        "Negative Prompt": args.np_prompt_path or args.np_prompt,
    }
    message = build_image_generation_start_message(args.model_name, model_info)
    slack_message_ts_start = slack_service.send_message(message=message, mention_id=mention_id)

    # if is_colab:
    if False: # Need quantization to avoid OOM when running EM with Turbo on A100 GPU
        # faster than quantization on A100 GPU
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    else:
        # WARN: OOM in 4bit model on 24GB TITAN RTX GPU
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        t5_nf4 = T5EncoderModel.from_pretrained(
            "diffusers/t5-nf4",
            torch_dtype=torch.bfloat16
        ).to("cuda")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_path,
            transformer=model_nf4,
            text_encoder_3=t5_nf4,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    new_data = []

    if args.image_output_dir:
        os.makedirs(args.image_output_dir, exist_ok=True)

    if args.np_prompt_path:
        with open(args.np_prompt_path, "r") as f:
            np_data = json.load(f)

        np_dict = {}
        for item in np_data:
            dict_image_id = item.get("image_id")
            dict_np = item.get("negative_prompt")
            if dict_image_id and dict_np:
                np_dict[dict_image_id] = dict_np
        print("np_dict len: ", len(np_dict))

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            image_id = f"{temp_piece['dataset_target']}_{temp_piece['image_id']}"
            normal_prompt = temp_piece["polished_prompt"]
            sp_positive_prompt = args.sp_prompt + "\n" + normal_prompt if args.sp_prompt and args.sp_num else normal_prompt

            neg_prompt = ""
            if args.np_prompt_path and args.np_num:
                neg_prompt = np_dict.get(image_id)
            elif args.np_prompt and args.np_num:
                neg_prompt = args.np_prompt

            if prompt_weighting:
                prompt = normal_prompt
                llm_prompt = None
                if args.sp_prompt and args.sp_num and args.sp_t5_only:
                    llm_prompt = sp_positive_prompt
                elif args.sp_prompt and args.sp_num:
                    prompt = sp_positive_prompt

                (prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sd3(
                    pipe,
                    prompt=prompt,
                    llm_prompt=llm_prompt,
                    neg_prompt=neg_prompt,
                    extend_clip=extend_clip,
                    extend_t5=exted_t5,
                    t5_max_length=max_sequence_length,
                )
                image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=prompt_neg_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    guidance_scale=guidance_scale,
                ).images[0]

                # image = pipe(
                #     prompt,
                #     negative_prompt=neg_prompt,
                #     height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                #     width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                #     num_inference_steps=num_inference_steps,
                #     max_sequence_length=max_sequence_length,
                #     guidance_scale=guidance_scale,
                # ).images[0]
            else:
                prompt = normal_prompt
                if args.sp_prompt and args.sp_num:
                    prompt = sp_positive_prompt

                image = pipe(
                    prompt,
                    height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    guidance_scale=guidance_scale,
                ).images[0]

            image_name = f"{temp_piece['dataset_target']}_{args.model_name}_{valid_image_count}_{temp_piece['image_id']}"
            image.save(os.path.join(args.image_output_dir, image_name))

            temp_json = {}
            temp_json["output_image_name"] = image_name
            temp_json["image_id"] = image_id
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

    # notify end
    message = build_image_generation_complete_message(args.model_name)
    slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
