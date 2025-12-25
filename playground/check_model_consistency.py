import os
import sys
import glob

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
GENERATED_JSON_DIR_PATH = "./outputs/image_info"
EVALUATION_MODEL_DIR_PATH = "./playground/evaluation"
# --------------------------------


def find_missing_models():
    """
    画像生成結果JSONには存在するが、モデルディレクトリが存在しないモデル名を出力
    """
    # 1. JSONファイルからモデル名のリストを取得
    json_pattern = os.path.join(GENERATED_JSON_DIR_PATH, "output_image_info_*.json")
    json_files = glob.glob(json_pattern)

    prefix = "output_image_info_"
    suffix = ".json"

    json_models = set()
    for file_path in json_files:
        filename = os.path.basename(file_path)

        # ファイル名からモデル名を抽出（プレフィックスとサフィックスを削除してモデル名を取得）
        if filename.startswith(prefix) and filename.endswith(suffix):
            model_name = filename[len(prefix):-len(suffix)]
            json_models.add(model_name)

    if not json_models:
        print(f"警告: ディレクトリ '{GENERATED_JSON_DIR_PATH}' 内で 'output_image_info_*.json' に一致するファイルが見つかりませんでした。")
        return []

    # print('json_models: ', json_models)

    # 2. モデルディレクトリのリストを取得
    dir_models = set()
    try:
        for item in os.listdir(EVALUATION_MODEL_DIR_PATH):
            item_path = os.path.join(EVALUATION_MODEL_DIR_PATH, item)

            if os.path.isdir(item_path):
                # '_average' が含まれていれば再検索
                if "_average" in item:
                    try:
                        for sub_item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_item_path):
                                dir_models.add(sub_item)
                    except Exception as e:
                        print(f"警告: '{item}' 内のディレクトリリスト取得中にエラー: {e}")
                else:
                    dir_models.add(item)

    except FileNotFoundError:
        print(f"エラー: モデルディレクトリ '{EVALUATION_MODEL_DIR_PATH}' が見つかりません。")
        return sorted(list(json_models))

    # print('dir_models: ', dir_models)

    # 3. 比較: JSONには存在するが、ディレクトリには存在しないモデル名をフィルタリング
    # setの差分演算子 (-) を利用して、一方にだけ存在する要素を抽出
    missing_models = sorted(list(json_models - dir_models))
    # print('missing_models: ', missing_models)

    return missing_models

if __name__ == "__main__":
    missing_list = find_missing_models()

    print("\n" + "#" * 50)
    if missing_list:
        print(f"✅ JSONファイルは存在するが、対応するモデルディレクトリが存在しないモデル ({len(missing_list)} 件):")
        for model in missing_list:
            print(f"- {model}")
    else:
        print("🎉 JSONファイルに対応する欠落したモデルディレクトリは見つかりませんでした。")
    print("#" * 50 + "\n")
