import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

from google import genai
from src.utils.env_detector import check_is_colab
from src.services.slack_client_service import slack_service, build_image_generation_start_message, build_image_generation_complete_message
import json
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', type=str, default="gemini-2.5-flash-lite") # or gemini-3-flash
    parser.add_argument('--prompt_path', type=str, default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    # parser.add_argument('--st_instruction_path', type=str, default=None)
    parser.add_argument('--st_instruction_path', type=str, default="./data-juicer/evaluation_pipeline/prompt_generation_example/LLM_ST_PROMPT_JP.md")


    args = parser.parse_args()

    args.output_json = f"./data-juicer/outputs/llm_prompt/output_st_info_{args.model_name}.json"

    return args


if __name__ == "__main__":
    args = parse_args()

    is_colab = check_is_colab()

    print(f"--- is_colab: {is_colab} ---")
    print("--- Model Name: ", args.model_name, " ---")
    print("--- Structure Prompt Instruction Path: ", args.st_instruction_path, " ---")

    # notify start
    mention_id = os.environ.get("SLACK_MENTION_ID")
    model_info = {
        "Structure Prompt Instruction Path": args.st_instruction_path,
    }
    message = build_image_generation_start_message(args.model_name, model_info)
    slack_message_ts_start = slack_service.send_message(message=message, mention_id=mention_id)

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()


    new_data = []

    processed_ids = set()
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            new_data = json.load(f)
            processed_ids = {item["image_id"] for item in new_data}
        print(f"--- Loaded {len(new_data)} existing records from {args.output_json} ---")

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    with open(args.st_instruction_path, "r") as f:
        instruction = f.read()

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            image_id = f"{temp_piece['dataset_target']}_{temp_piece['image_id']}"
            positive_prompt = temp_piece["polished_prompt"]

            prompt = f"""{instruction}
{positive_prompt}
"""

            response = client.models.generate_content(
                model=args.model_name, contents=prompt
            )
            generated_text = response.text

            temp_json = {}
            temp_json["image_id"] = image_id
            temp_json["llm_output"] = generated_text
            new_data.append(temp_json)

            valid_image_count += 1

        except Exception as e:
            print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
            print(e)
            print("----------------------------------------------------------")
            message = f":warning: ({valid_image_count})回目でエラーが発生しました: {e} for prompt ID {temp_piece['image_id']}"
            slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
            valid_image_count += 1
        # except:
        #     continue

    with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    # notify end
    message = build_image_generation_complete_message(args.model_name)
    slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
