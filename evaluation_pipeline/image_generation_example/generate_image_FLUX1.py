import os
import torch
from diffusers import DiffusionPipeline
import json
import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="SD1_5")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    parser.add_argument('--icl_num', type=str, default=None)
    parser.add_argument('--icl_prompt', type=str, default=None)

    args = parser.parse_args()

    if 'dev' in args.model_name:
        args.model_path = "black-forest-labs/FLUX.1-dev"
    elif 'schnell' in args.model_name:
        args.model_path = "black-forest-labs/FLUX.1-schnell"

    args.output_json = f"./evaluation_pipeline/image_generation_example/output_image_info_{args.model_name}.json"
    args.image_output_dir = f"./evaluation_pipeline/image_generation_example/output_image_{args.model_name}/"

    return args


if __name__ == "__main__":
    args = parse_args()

    # --- Colab Configuration ---
    try:
        import google.colab
        is_colab = True
        DRIVE_PATH_BASE = '/content/drive/MyDrive/workspace//huggingface_cache/'
        cache_directory = DRIVE_PATH_BASE
    except:
        is_colab = False
        cache_directory = None
    print(f"--- is_colab: {is_colab} ---")
    # ----------------------------

    # GPU節約のためにto("cuda")の移行をなしにすると遅いので移行
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        # torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        cache_directory=cache_directory,
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
            # print(f"--- before prompt {prompt} ---")
            if args.icl_prompt and args.icl_num:
                prompt = args.icl_prompt + "\n" + prompt
            # print(f"--- after prompt {prompt} ---")
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
