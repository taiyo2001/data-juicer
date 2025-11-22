import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "data_juicer/evaluation_pipeline"))
# --------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from extract_prompt import extract_boxed_prompt
import torch
import json
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-14B")
    parser.add_argument('--model_name', type=str, default="Qwen3-14B")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--gguf_model', type=str, default=None)
    # negative prompt
    parser.add_argument('--np_instruction_path', type=str, default=None)

    args = parser.parse_args()

    # Qwen3-14B
    if args.model_name == "Qwen3-14B":
        args.model_path = "Qwen/Qwen3-14B"
    elif args.model_name == "Qwen3-14B-FP8":
        args.model_path = "Qwen/Qwen3-14B-FP8"
    elif args.model_name == "Qwen3-14B-GGUF":
        args.model_path = "Qwen/Qwen3-14B-GGUF"
        args.gguf_model = "Qwen3-14B-Q8_0.gguf"
    elif args.model_name == "Qwen3-30B-A3B-Thinking-2507-GGUF":
        args.model_path = "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
        # args.gguf_model = "Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf"
        args.gguf_model = "Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf"


    args.output_json = f"./evaluation_pipeline/prompt_generation_example/output_np_info_{args.model_name}.json"

    return args


if __name__ == "__main__":
    args = parse_args()

    is_gguf = False
    if "-GGUF" in args.model_name:
        is_gguf = True

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
    print("--- GGUF: ", is_gguf, args.gguf_model, " ---")


    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU may be very slow for large models.")
        exit(1)

    if is_gguf and args.gguf_model is not None:
        llm = Llama.from_pretrained(
            repo_id=args.model_path,
            filename=args.gguf_model,
            n_ctx=8192,
            n_gpu_layers=-1
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path
            # torch_dtype="auto",
            # device_map="auto"
        )

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

            if is_gguf:
                output = llm.create_completion(
                    prompt,
                    # max_tokens=128,
                    max_tokens=32768,
                    # temperature=0.1,
                )
                generated_text = output['choices'][0]['text']
            else:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                generated_text = model.generate(**inputs, max_new_tokens=128)[0][inputs["input_ids"].shape[-1]:]

            extract_boxed_prompted = extract_boxed_prompt(generated_text)

            if extract_boxed_prompted is None:
                extract_boxed_prompted = ""

            temp_json = {}
            temp_json["image_id"] = temp_piece["dataset_target"] + "_" + temp_piece["image_id"]
            temp_json["llm_output"] = generated_text
            temp_json["negative_prompt"] = extract_boxed_prompted
            new_data.append(temp_json)

            valid_image_count += 1

        except Exception as e:
            print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
            print(e)
            print("----------------------------------------------------------")
        # except:
        #     continue

    with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
