import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

import torch
from diffusers import DiffusionPipeline
from sd_embed.src.sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15
from src.services.slack_client_service import slack_service, build_image_generation_start_message, build_image_generation_complete_message
from src.utils.env_detector import check_is_colab
from src.constants import DETAIL_MASTER
import json
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument('--model_name', type=str, default="SD1_5")
    parser.add_argument('--prompt_path', type=str, default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    parser.add_argument('--semantic_threshold', type=int, default=None)
    parser.add_argument('--count', type=str, default=None)
    parser.add_argument('--save_start_threshold', type=int, default=None)
    # in-context learning
    parser.add_argument('--sp_num', type=str, default=None)
    parser.add_argument('--sp_prompt', type=str, default=None)
    # negative prompt
    parser.add_argument('--np_num', type=str, default=None)
    parser.add_argument('--np_prompt', type=str, default=None)
    parser.add_argument('--np_prompt_path', type=str, default=None)
    # dense prompt
    parser.add_argument('--dp_num', type=str, default=None)
    parser.add_argument('--dp_prompt_path', type=str, default=None)

    args=parser.parse_args()

    if args.semantic_threshold is not None:
        args.model_name = args.model_name + f"-{args.semantic_threshold}"

    # system prompt
    if args.sp_num is not None and args.sp_prompt is not None:
        args.model_name = args.model_name + f"_SP{args.sp_num}"

    # negative prompt
    if args.np_num is not None and (args.np_prompt is not None or args.np_prompt_path is not None):
        args.model_name = args.model_name + f"_NP{args.np_num}"

    # dense prompt
    if args.dp_num is not None and args.dp_prompt_path is not None:
        args.model_name = args.model_name + f"_DP{args.dp_num}"

    if args.count is not None:
        args.model_name = args.model_name + f"_{args.count}"

    args.output_json = f"./data-juicer/outputs/image_info/output_image_info_{args.model_name}.json"
    args.image_output_dir = f"./data-juicer/outputs/image/output_image_{args.model_name}/"

    return args


if __name__ == "__main__":
    args = parse_args()

    prompt_weighting = False
    chunking_method = None
    if "_EM" in args.model_name:
        prompt_weighting = True
        if "_EM-SE" in args.model_name:
            chunking_method = "sentence"
            sentence_chunking = True
        elif "_EM-S" in args.model_name and args.semantic_threshold is not None:
            chunking_method = "semantic"

    is_colab = check_is_colab()

    print(f"--- is_colab: {is_colab} ---")
    print("--- Model Name: ", args.model_name, " ---")
    print("--- Prompt Weighting: ", prompt_weighting, "Semantic Chunking: ", chunking_method, args.semantic_threshold, " ---")
    print("--- SP Num: ", args.sp_num, " ---")
    print("--- SP Prompt: ", args.sp_prompt, " ---")
    print("--- Negative Prompt Num: ", args.np_num, " ---")
    print("--- Negative Prompt: ", args.np_prompt_path or args.np_prompt, " ---")
    print("--- Dense Prompt Num: ", args.dp_num, " ---")
    print("--- Dense Prompt: ", args.dp_prompt_path, " ---")
    print("--- SAVE START THRESHOLD: ", args.save_start_threshold, " ---")

    # notify start
    mention_id = os.environ.get("SLACK_MENTION_ID")
    model_info = {
        "Prompt Weighting": prompt_weighting,
        "Chunking Method": chunking_method,
        "SP Num": args.sp_num,
        "SP Prompt": args.sp_prompt,
        "Negative Prompt Num": args.np_num,
        "Negative Prompt": args.np_prompt_path or args.np_prompt,
        "Dense Prompt Num": args.dp_num,
        "Dense Prompt": args.dp_prompt_path,
        "SAVE START THRESHOLD": args.save_start_threshold,
    }
    message = build_image_generation_start_message(args.model_name, model_info)
    slack_message_ts_start = slack_service.send_message(message=message, mention_id=mention_id)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
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

    if args.dp_prompt_path:
        with open(args.dp_prompt_path, "r") as f:
            dp_data = json.load(f)

        dp_dict = {}
        for item in dp_data:
            dict_image_id = item.get("image_id")
            dict_dp = item.get("llm_output")
            if dict_image_id and dict_dp:
                dp_dict[dict_image_id] = dict_dp
        print("dp_dict len: ", len(dp_dict))


    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    total_count = len(data)
    report_step = max(1, total_count // 10)
    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            image_id = f"{temp_piece['dataset_target']}_{temp_piece['image_id']}"
            image_name = f"{temp_piece['dataset_target']}_{args.model_name}_{valid_image_count}_{temp_piece['image_id']}"
            normal_prompt = temp_piece["polished_prompt"]
            sp_positive_prompt = args.sp_prompt + "\n" + normal_prompt if args.sp_prompt and args.sp_num else normal_prompt

            neg_prompt = ""
            if args.np_prompt_path and args.np_num:
                neg_prompt = np_dict.get(image_id)
            elif args.np_prompt and args.np_num:
                neg_prompt = args.np_prompt

            dense_prompt = None
            if args.dp_prompt_path and args.dp_num:
                dense_prompt = dp_dict.get(image_id)
                if dense_prompt is not None:
                    print("Dense Prompt: ", dense_prompt[:50], "...")
                else:
                    print("No Dense Prompt found for ", image_id)

            if args.save_start_threshold is None or valid_image_count > args.save_start_threshold:
                if prompt_weighting:
                    # normal
                    prompt = normal_prompt
                    # -- SP ---
                    if args.sp_prompt and args.sp_num:
                        prompt = sp_positive_prompt
                    # -- DP ---
                    if args.dp_prompt_path and args.dp_num and dense_prompt is not None:
                        print("Adapt Dense Prompt")
                        prompt = dense_prompt

                    (prompt_embeds, prompt_neg_embeds) = get_weighted_text_embeddings_sd15(
                        pipe,
                        prompt=prompt,
                        neg_prompt=neg_prompt,
                        chunking_method=chunking_method,
                        semantic_threshold=args.semantic_threshold
                    )
                    image = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=prompt_neg_embeds,
                        height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                        width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                        num_inference_steps=30,
                    ).images[0]
                else:
                    prompt = normal_prompt
                    if args.sp_prompt and args.sp_num:
                        prompt = sp_positive_prompt

                    # TODO: GuidanceScale Adaption
                    image = pipe(
                        prompt,
                        height=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                        width=DETAIL_MASTER.IMAGE_SIZE.SMALL,
                        num_inference_steps=50,
                    ).images[0]

                image.save(os.path.join(args.image_output_dir, image_name))
            else:
                print(f"Skipping image save for index {valid_image_count} (JSON entry only)")

            temp_json = {}
            temp_json["output_image_name"] = image_name
            temp_json["image_id"] = image_id
            new_data.append(temp_json)

            valid_image_count += 1

            # --- noti progress ---
            if valid_image_count % report_step == 0 and valid_image_count < total_count:
                percentage = (valid_image_count / total_count) * 100
                progress_message = (
                    f" :hourglass_flowing_sand: 進捗報告: {percentage:.0f}% 完了 "
                    f"({valid_image_count}/{total_count})\n"
                )
                slack_service.send_message(message=progress_message, thread_ts=slack_message_ts_start)

        except Exception as e:
            print(f"\n--- ERROR encountered for prompt {temp_piece['image_id']} ---")
            print(e)
            print("----------------------------------------------------------")
            message = f":warning: ({valid_image_count})回目でエラーが発生しました: {e} for prompt ID {temp_piece['image_id']}"
            slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
            valid_image_count += 1
        # except:
        #     continue

    with open(args.output_json, "a") as f:
        json.dump(new_data, f)

    # notify end
    message = build_image_generation_complete_message(args.model_name)
    slack_message_ts_complete = slack_service.send_message(message=message, thread_ts=slack_message_ts_start)
