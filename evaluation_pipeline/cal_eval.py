import json
import os
import argparse

DEFAULT_EVAL_OUTPUT_LOG_ROOT = "./data-juicer/playground/evaluation"
EXCLUDED_DIRS = ['comparison_results', 'excluded_model_name']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_output_log_dir_name', type=str, default=None)
    parser.add_argument('--name_prefix', type=str, default=None)
    args=parser.parse_args()

    return args

def cal_eval(dir_name, name_prefix):
    file_name = os.listdir(dir_name)
    overall_eval_file_name = ""

    print('model name: ', name_prefix)

    object_presence_all_count = 0
    object_presence_success_count = 0
    character_attributes_all_count = {"animal":0, "object":0, "person":0}
    character_attributes_success_count = {"animal":0, "object":0, "person":0}
    character_locations_all_count = 0
    character_locations_success_count = 0
    scene_attrbutes_all_count = {"background":0, "light":0, "style":0, "spatial":0}
    scene_attrbutes_success_count = {"background":0, "light":0, "style":0, "spatial":0}

    for temp_name in file_name:
        if name_prefix in temp_name and "overall" in temp_name:
            overall_eval_file_name = temp_name
            with open(os.path.join(dir_name, temp_name), "r") as f:
                temp_json = json.load(f)
                object_presence_all_count += temp_json["object_presence_all_count"]
                object_presence_success_count += temp_json["object_presence_success_count"]
                character_attributes_all_count["animal"] += temp_json["character_attributes_all_count"]["animal"]
                character_attributes_all_count["object"] += temp_json["character_attributes_all_count"]["object"]
                character_attributes_all_count["person"] += temp_json["character_attributes_all_count"]["person"]
                character_attributes_success_count["animal"] += temp_json["character_attributes_success_count"]["animal"]
                character_attributes_success_count["object"] += temp_json["character_attributes_success_count"]["object"]
                character_attributes_success_count["person"] += temp_json["character_attributes_success_count"]["person"]
                character_locations_all_count += temp_json["character_locations_all_count"]
                character_locations_success_count += temp_json["character_locations_success_count"]
                scene_attrbutes_all_count["background"] += temp_json["scene_attrbutes_all_count"]["background"]
                scene_attrbutes_all_count["light"] += temp_json["scene_attrbutes_all_count"]["light"]
                scene_attrbutes_all_count["style"] += temp_json["scene_attrbutes_all_count"]["style"]
                scene_attrbutes_all_count["spatial"] += temp_json["scene_attrbutes_all_count"]["spatial"]
                scene_attrbutes_success_count["background"] += temp_json["scene_attrbutes_success_count"]["background"]
                scene_attrbutes_success_count["light"] += temp_json["scene_attrbutes_success_count"]["light"]
                scene_attrbutes_success_count["style"] += temp_json["scene_attrbutes_success_count"]["style"]
                scene_attrbutes_success_count["spatial"] += temp_json["scene_attrbutes_success_count"]["spatial"]

    acc = {}
    acc["object_presence"] = round(object_presence_success_count/object_presence_all_count, 4)
    acc["character_attributes_animal"] = round(character_attributes_success_count["animal"]/character_attributes_all_count["animal"], 4)
    acc["character_attributes_object"] = round(character_attributes_success_count["object"]/character_attributes_all_count["object"], 4)
    acc["character_attributes_person"] = round(character_attributes_success_count["person"]/character_attributes_all_count["person"], 4)
    acc["character_locations"] = round(character_locations_success_count/character_locations_all_count, 4)
    acc["scene_attrbutes_background"] = round(scene_attrbutes_success_count["background"]/scene_attrbutes_all_count["background"], 4)
    acc["scene_attrbutes_light"] = round(scene_attrbutes_success_count["light"]/scene_attrbutes_all_count["light"], 4)
    acc["scene_attrbutes_style"] = round(scene_attrbutes_success_count["style"]/scene_attrbutes_all_count["style"], 4)
    acc["scene_attrbutes_spatial"] = round(scene_attrbutes_success_count["spatial"]/scene_attrbutes_all_count["spatial"], 4)


    print(acc)

    output_file_name = overall_eval_file_name.replace("_overall_eval.json", "_final_score.json")
    output_filepath = os.path.join(dir_name, output_file_name)

    return acc, output_filepath


def save_eval_result(output_filepath, acc):
    with open(output_filepath, "w") as f:
        json.dump(acc, f, indent=4)

    print("-" * 30)
    print(f"✅ 最終結果が保存されました: {output_filepath}")
    print("-" * 30)


if __name__ == "__main__":
    args = parse_args()

    # 引数未指定であれば、デフォルトディレクトリ内の全フォルダを処理
    if args.eval_output_log_dir_name is None or args.name_prefix is None:
        log_dirs = os.listdir(DEFAULT_EVAL_OUTPUT_LOG_ROOT)

        for item_name in log_dirs:
            dir_path = os.path.join(DEFAULT_EVAL_OUTPUT_LOG_ROOT, item_name)

            if not os.path.isdir(dir_path):
                continue

            if item_name in EXCLUDED_DIRS:
                continue

            if item_name.endswith('_average'):
                continue

            dir_name = dir_path
            name_prefix = item_name
            acc, output_filepath = cal_eval(dir_name, name_prefix)
            save_eval_result(output_filepath, acc)
    else:
        dir_name = args.eval_output_log_dir_name
        name_prefix = args.name_prefix
        acc, output_filepath = cal_eval(dir_name, name_prefix)
        save_eval_result(output_filepath, acc)
