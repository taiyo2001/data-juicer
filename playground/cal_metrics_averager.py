# Before Exec:
# 1. mkdir {models}_average
# 2. python python evaluation_pipeline/cal_eval.py (generate final evaluation metrics)
# 3. mv {models} {models}_average/{model_name}

import os
import sys
import json
import glob
from collections import defaultdict

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
EVALUATION_ROOT = os.path.join(PROJECT_ROOT, "data-juicer/playground/evaluation/")
# --------------------------------

ARR_MODELS_LIST = [
    'FLUX1-schnell_ICL1',
    # 'FLUX1-schnell'
    'SD1_5_EM',
    'SD1_5_EM_ICL1',
    # 'SD1_5_EM_ICL2',
    'SD1_5_EM_NP1',
    'ParaDiffusion_L',
    'ParaDiffusion_L_ICL1',
    'ParaDiffusion_L_ICL2',
]


def get_target_models_from_average_dir(group_name):
    """
    '{group_name}_average' ディレクトリ内のサブディレクトリをターゲットモデルとして取得
    """
    avg_dir_name = f"{group_name}_average"
    avg_dir_path = os.path.join(EVALUATION_ROOT, avg_dir_name)

    if not os.path.isdir(avg_dir_path):
        print(f"WARN: 平均スコアディレクトリ '{avg_dir_path}' が見つかりません。")
        return []

    target_models = []
    try:
        for item in os.listdir(avg_dir_path):
            item_path = os.path.join(avg_dir_path, item)
            if os.path.isdir(item_path):
                target_models.append(item)
    except Exception as e:
        print(f"エラー: '{avg_dir_path}' 内のディレクトリリスト取得中にエラー: {e}")
        return []

    return target_models


def calculate_average_min_max_range_scores(group_name, target_models):
    """評価指標の平均・最大・最小・差の計算"""

    metric_sums = defaultdict(float)
    metric_mins = {}
    metric_maxs = {}

    processed_count = 0
    all_json_files = []
    processed_model_names = set()

    avg_group_name = f"{group_name}_average"

    for model_name in target_models:
        search_pattern = os.path.join(EVALUATION_ROOT, avg_group_name, model_name, "*_final_score.json")
        found_files = glob.glob(search_pattern)

        if not found_files:
            print(f"警告: モデル '{model_name}' のディレクトリ内で '*_final_score.json' に一致するファイルが見つかりません。")
            continue

        all_json_files.extend(found_files)
        processed_model_names.add(model_name)

    for json_path in all_json_files:
        try:
            with open(json_path, 'r') as f:
                scores = json.load(f)

            file_basename = os.path.basename(json_path)

            # スコア集計と最小/最大の更新
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] += value

                    if key not in metric_mins:
                        metric_mins[key] = value
                        metric_maxs[key] = value
                    else:
                        metric_mins[key] = min(metric_mins[key], value)
                        metric_maxs[key] = max(metric_maxs[key], value)
                else:
                    print(f"警告: ファイル '{file_basename}' の '{key}' の値が数値ではありません。スキップします。")

            processed_count += 1
            print(f"成功: '{file_basename}'")

        except json.JSONDecodeError:
            print(f"エラー: {json_path} のJSON形式が不正です。スキップします。")
        except Exception as e:
            print(f"予期せぬエラー ({json_path}): {e}")

    if processed_count == 0:
        print(f"エラー: 有効なファイルが一つも見つからなかったため、平均を計算できません。")
        return None

    average_scores = {}
    min_max_scores = {}

    min_max_scores["processed_models"] = sorted(list(processed_model_names))

    for key, total_sum in metric_sums.items():
        # 平均計算
        average_score = total_sum / processed_count

        formatted_str = f"{average_score:.4g}"
        average_scores[key] = float(formatted_str)

        # 最大値・最小値・差の計算
        min_val = metric_mins[key]
        max_val = metric_maxs[key]
        range_val = max_val - min_val

        min_max_scores[key] = {
            'min': float(f"{min_val:.4g}"),
            'max': float(f"{max_val:.4g}"),
            'range': float(f"{range_val:.4g}")
        }

    print(f"--- 完了 --- (平均対象ファイル数: {processed_count} 個)")
    return average_scores, min_max_scores


def save_scores(path, filename, scores):
    os.makedirs(path, exist_ok=True)

    output_file = os.path.join(path, filename)

    try:
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)
        print(f"✅ 結果を保存: {output_file}")
    except Exception as e:
        print(f"❌ 保存中にエラーが発生: {e}")


if __name__ == "__main__":
    for group_name in ARR_MODELS_LIST:
        target_models = get_target_models_from_average_dir(group_name)

        if not target_models:
            print(f"スキップ: グループ '{group_name}' の平均対象モデルが見つかりませんでした。")
            continue

        avg_scores, min_max_scores = calculate_average_min_max_range_scores(group_name, target_models)

        if avg_scores:
            avg_group_name = f"{group_name}_average"
            path = os.path.join(EVALUATION_ROOT, avg_group_name)
            avg_scores_filename = f"{group_name}_average_score.json"
            min_max_scores_filename = f"{group_name}_min_max_score.json"

            save_scores(path, avg_scores_filename, avg_scores)
            save_scores(path, min_max_scores_filename, min_max_scores)
