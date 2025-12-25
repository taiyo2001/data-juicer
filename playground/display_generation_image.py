import argparse
import json
import os
from PIL import Image

IMAGE_DIR_PREFIX = "./outputs/image/output_image_"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="SD1_5_SP1")
    parser.add_argument('--image_id', type=str, default="qual_dev_00003.jpg")
    parser.add_argument('--prompt_path', type=str, default="./DetailMaster_Dataset/DetailMaster_Dataset.json")

    args = parser.parse_args()
    return args

def get_image_info(args):

    with open(args.prompt_path, 'r') as f:
            data = json.load(f)

    for entry in data:
        if entry.get("image_id") == args.image_id:
            return entry

    print(f"エラー: データセット内に image_id '{args.image_id}' に対応するエントリが見つかりませんでした。")
    return None

def display_result(args):
    image_dir = f"{IMAGE_DIR_PREFIX}{args.model_name}"

    info = get_image_info(args)
    if not info:
        return

    # ディレクトリ内のファイルを検索
    found_filename = None
    for filename in os.listdir(image_dir):
        # ファイル名が画像IDで終わっているものを探す (例: docci_1_qual_dev_00003.jpg.png)
        if args.image_id in filename and filename.endswith((".jpg", ".png")):
             found_filename = filename
             break

    if not found_filename:
        print(f"エラー: ディレクトリ '{image_dir}' 内に画像ID '{args.image_id}' を含む画像ファイルが見つかりませんでした。")
        return

    image_path = os.path.join(image_dir, found_filename)

    print("-" * 50)
    print(f"✅ 画像生成結果の確認 - モデル: {args.model_name}")
    print(f"📂 画像ファイルパス: {image_path}")
    print(f"🆔 画像ID: {args.image_id}")
    print("-" * 50)

    print("📝 Polished Prompt:")
    print(info["polished_prompt"])
    print("\n")

    img = Image.open(image_path)
    print(f"🖼️ 画像サイズ: {img.size}")
    return img


if __name__ == "__main__":
    args = parse_args()

    img = display_result(args)
