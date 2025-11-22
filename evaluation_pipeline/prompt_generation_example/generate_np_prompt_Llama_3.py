import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument('--model_name', type=str, default="Llama-3_1")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--use_quantization', type=bool, default=False)
    # negative prompt
    parser.add_argument('--np_instruction_path', type=str, default=None)

    args = parser.parse_args()

    if args.model_name == "Llama-3_1":
        args.model_path = "meta-llama/Llama-3.1-8B"
    elif args.model_name == "Llama-3_3":
        # WARN: OOM in 4bit model on A100 GPU
        args.model_path = "meta-llama/Llama-3.3-70B-Instruct"

    args.output_json = f"./evaluation_pipeline/prompt_generation_example/output_np_info_{args.model_name}.json"

    return args


if __name__ == "__main__":
    args = parse_args()

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
    print("--- Negative Prompt Instruction Path: ", args.np_instruction_path, " ---")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU may be very slow for large models.")
        exit(1)

    quantization_config = None
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # llm_int8_enable_fp32_cpu_offload=True
            load_in_4bit=True,
        )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    new_data = []

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    with open(args.np_instruction_path, "r") as f:
        instruction = f.read()

    if "_EN" in args.np_instruction_path:
        task_title = 'Generate Negative Prompt'
    elif "_JP" in args.np_instruction_path:
        task_title = '生成すべきネガティブプロンプト'

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            positive_prompt = temp_piece["polished_prompt"]

            prompt = f"""{instruction}
{positive_prompt}

**{task_title}:**
"""

            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tensor = quantized_model.generate(**input_ids, max_new_tokens=128)
            generated_token_ids = output_tensor[0][input_ids["input_ids"].shape[-1]:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            # print(f"generated_text: {generated_text}")

            temp_json = {}
            temp_json["image_id"] = temp_piece["dataset_target"] + "_" + temp_piece["image_id"]
            temp_json["negative_prompt"] = generated_text
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
