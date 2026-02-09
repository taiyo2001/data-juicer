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
import time
import json
import tqdm
import argparse

# START_INDEX = 2000

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', type=str, default="gemini-3-flash-preview")
    parser.add_argument('--prompt_path', type=str, default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--prompt_type', type=str, default=None)
    parser.add_argument('--instruction_name', type=str, default=None)
    parser.add_argument('--instruction_path', type=str, default=None)

    args = parser.parse_args()

    if args.prompt_type is None or args.instruction_name is None:
        raise ValueError("Both --prompt_type and --instruction_name must be provided.")

    args.instruction_path = f"./data-juicer/evaluation_pipeline/prompt_generation_example/{args.instruction_name}"
    args.output_json = f"./data-juicer/outputs/llm_prompt/output_{args.prompt_type}_info_{args.model_name}.json"
    # args.output_json = f"./data-juicer/outputs/llm_prompt/output_{args.prompt_type}_info_{args.model_name}_{START_INDEX}-END.json"

    return args


if __name__ == "__main__":
    args = parse_args()

    is_colab = check_is_colab()

    print(f"--- is_colab: {is_colab} ---")
    print("--- Model Name: ", args.model_name, " ---")
    print("--- Instruction Path: ", args.instruction_path, " ---")

    # notify start
    mention_id = os.environ.get("SLACK_MENTION_ID")
    model_info = {
        "Instruction Path": args.instruction_path,
        "Output Json": args.output_json,
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

    data_dict = {item["image_id"]: item.get("llm_output") for item in new_data}

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    with open(args.instruction_path, "r") as f:
        instruction = f.read()

    total_count = len(data)
    report_step = max(1, total_count // 10)
    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
    # 開始位置調整
    # for temp_piece in tqdm.tqdm(data[:START_INDEX], total=START_INDEX): # 最初〜START_INDEX
    # for temp_piece in tqdm.tqdm(data[START_INDEX:], total=len(data) - START_INDEX): # START_INDEX〜最後
        image_id = f"{temp_piece['dataset_target']}_{temp_piece['image_id']}"
        positive_prompt = temp_piece["polished_prompt"]
        prompt = f"{instruction}\n{positive_prompt}\n"

        max_retries = 3
        retry_count = 0
        success = False

        existing_output = data_dict.get(image_id)
        if existing_output is not None and existing_output != "":
            valid_image_count += 1
            continue
        # if image_id in processed_ids:
        #     valid_image_count += 1
        #     continue

        while retry_count <= max_retries and not success:
            try:
                response = client.models.generate_content(
                    model=args.model_name, contents=prompt
                )
                generated_text = response.text
                print(generated_text)

                # Update if exists, otherwise append.
                target_item = next((item for item in new_data if item["image_id"] == image_id), None)
                if target_item:
                    target_item["llm_output"] = generated_text
                else:
                    temp_json = {
                        "image_id": image_id,
                        "llm_output": generated_text
                    }
                    new_data.append(temp_json)

                valid_image_count += 1
                success = True

                # --- noti progress ---
                if valid_image_count % report_step == 0 and valid_image_count < total_count:
                    percentage = (valid_image_count / total_count) * 100
                    progress_message = (
                        f" :hourglass_flowing_sand: 進捗報告: {percentage:.0f}% 完了 "
                        f"({valid_image_count}/{total_count})\n"
                    )
                    slack_service.send_message(message=progress_message, thread_ts=slack_message_ts_start)

            except Exception as e:
                # retry: 503（Service Unavailable）
                error_str = str(e)
                if "503" in error_str or "overloaded" in error_str.lower():
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = retry_count * 3 # リトライごとに待機時間増加
                        print(f"\n--- Model overloaded (503). Retrying {retry_count}/{max_retries} after {wait_time}s ---")
                        time.sleep(wait_time)
                        continue

                print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
                print(e)
                print("----------------------------------------------------------")
                message = f":warning: ({valid_image_count})回目でエラーが発生し、中断しました: {e} for prompt ID {temp_piece['image_id']}"
                slack_service.send_message(message=message, thread_ts=slack_message_ts_start)

                valid_image_count += 1
                break
            # except:
            #     continue

    with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    # notify end
    message = build_image_generation_complete_message(args.model_name)
    slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
