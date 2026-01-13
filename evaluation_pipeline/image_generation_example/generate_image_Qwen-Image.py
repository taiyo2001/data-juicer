import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from src.services.slack_client_service import slack_service, build_image_generation_start_message, build_image_generation_complete_message
from src.utils.env_detector import check_is_colab
from src.constants import DETAIL_MASTER
import torch
import math
import json
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-Image")
    parser.add_argument('--model_name', type=str, default="Qwen-Image-Lightning")
    parser.add_argument('--lora_path', type=str, default="lightx2v/Qwen-Image-Lightning")
    parser.add_argument('--lora_name', type=str, default="Qwen-Image-Lightning-4steps-V2.0.safetensors")
    parser.add_argument('--prompt_path', type=str, default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    parser.add_argument('--count', type=str, default=None)
    # in-context learning
    parser.add_argument('--sp_num', type=str, default=None)
    parser.add_argument('--sp_prompt', type=str, default=None)
    # negative prompt
    parser.add_argument('--np_num', type=str, default=None)
    parser.add_argument('--np_prompt', type=str, default=None)
    parser.add_argument('--np_prompt_path', type=str, default=None)

    args=parser.parse_args()

    if args.sp_num is not None and args.sp_prompt is not None:
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

    num_inference_steps = 50
    if 'Lightning' in args.model_name:
        num_inference_steps = 4

    max_sequence_length = 512 # default
    if '_SL512' in args.model_name:
        max_sequence_length = 512
    elif '_SL1024' in args.model_name:
        max_sequence_length = 1024

    is_colab = check_is_colab()

    print(f"--- is_colab: {is_colab} ---")
    print("--- Model Name: ", args.model_name, " ---")
    print("--- Max Sequence Length: ", max_sequence_length, " ---")
    print("--- SP Num: ", args.sp_num, " ---")
    print("--- SP Prompt: ", args.sp_prompt, " ---")
    print("--- Negative Prompt Num: ", args.np_num, " ---")
    print("--- Negative Prompt: ", args.np_prompt_path or args.np_prompt, " ---")

    # notify start
    mention_id = os.environ.get("SLACK_MENTION_ID")
    model_info = {
        "SP Num": args.sp_num,
        "SP Prompt": args.sp_prompt,
        "Negative Prompt Num": args.np_num,
        "Negative Prompt": args.np_prompt_path or args.np_prompt,
    }
    message = build_image_generation_start_message(args.model_name, model_info)
    slack_message_ts_start = slack_service.send_message(message=message, mention_id=mention_id)


    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    quantization_config = PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            ),
            "text_encoder": TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    )
    # QwenImagePipeline is registered as DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    pipe.load_lora_weights(
        args.lora_path, weight_name=args.lora_name
    )
    pipe.to("cuda")


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

            prompt = normal_prompt
            if args.sp_prompt and args.sp_num:
                prompt = sp_positive_prompt

            image = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=DETAIL_MASTER.IMAGE_SIZE.LARGE,
                height=DETAIL_MASTER.IMAGE_SIZE.LARGE,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=1.0,
                max_sequence_length=max_sequence_length, # default 512
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
