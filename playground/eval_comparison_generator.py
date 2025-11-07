import os
import sys
import pandas as pd
import json
import glob
import argparse
import matplotlib.pyplot as plt

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
EVALUATION_ROOT = os.path.join(PROJECT_ROOT, "data-juicer/playground/evaluation/")
COMPARISON_RESULT_ROOT = os.path.join(PROJECT_ROOT, "data-juicer/playground/evaluation/comparison_results/")
# --------------------------------

MODELS_TO_COMPARE = [
    {"base_model": "FLUX1-schnell", "icl_num": 1},
    {"base_model": "FLUX1-schnell", "icl_num": 2},
    {"base_model": "ParaDiffusion", "icl_num": 1},
    {"base_model": "ParaDiffusion", "icl_num": 2},
    {"base_model": "ParaDiffusion_L", "icl_num": 1},
    {"base_model": "ParaDiffusion_L", "icl_num": 2},
    {"base_model": "SD1_5", "icl_num": 1},
    {"base_model": "SD1_5", "icl_num": 2},
    {"base_model": "SD1_5_EM", "icl_num": 1},
    {"base_model": "SD1_5_EM", "icl_num": 2},
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", type=str, nargs='+', default=None)
    args=parser.parse_args()

    return args


def find_final_score_file(model_dir):
    search_path = os.path.join(model_dir, "*_final_score.json")
    files = glob.glob(search_path)
    if files:
        return files[0]
    return None


def load_score(filepath):
    """JSONファイルから評価スコアを読み込む"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: ファイルが見つかりません: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"エラー: JSON形式が不正です: {filepath}")
        return None


def format_index(index):
    """Pandasのインデックス名を整形"""
    return index.replace('_', ' ').title().replace('Attrbutes', 'Attributes')


def save_table_as_image(df, model_1, model_2):
    """DataFrameの画像保存"""
    image_output_name = f"{model_1}_vs_{model_2}_comparison_summary.png"
    filepath = os.path.join(COMPARISON_RESULT_ROOT, image_output_name)
    title = f"Evaluation Comparison: {model_1} vs {model_2}"

    col_name_1_score = f'{model_1} Score'
    col_name_2_score = f'{model_2} Score'
    change_col_name = 'Percent Change'

    png_filepath = filepath.replace(".csv", ".png")

    df_display = df.copy()

    def format_pct_diff(row):
            pct_change = row[change_col_name] * 100
            sign = '▲' if pct_change > 0.001 else ('▼' if pct_change < -0.001 else '—')
            return f"{pct_change:+.2f}% ({sign})"

    change_values = df_display[change_col_name]

    df_display[col_name_1_score] = (df_display[model_1] * 100).round(2).astype(str) + '%'
    df_display[col_name_2_score] = (df_display[model_2] * 100).round(2).astype(str) + '%'
    df_display['Change'] = df_display.apply(format_pct_diff, axis=1)

    df_final = df_display[[col_name_1_score, col_name_2_score, 'Change']]

    num_rows = len(df_final) + 1
    fig_height = max(3.0, num_rows * 1.0)

    # fig, ax = plt.subplots(figsize=(12, 7))
    fig, ax = plt.subplots(figsize=(10, fig_height)) # (横幅, 縦幅)

    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df_final.values,
                     colLabels=df_final.columns,
                     rowLabels=df_final.index,
                     cellLoc='center',
                     loc='top'
                     )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 色分け
    num_rows = len(df_final)
    change_col_index = 2 # 'Change' カラムのインデックス (0, 1, 2)

    for i in range(num_rows):
        change_val = change_values.iloc[i]

        if change_val > 0.001:
            color = '#87CEEB'
        elif change_val < -0.001:
            color = '#FFA07A'
        else:
            color = 'lightgray'

        cell = table.get_celld()[i + 1, change_col_index]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold')

    plt.title(title, y=0.90, fontsize=14, loc='center')
    plt.tight_layout()

    plt.savefig(png_filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return png_filepath


def format_output_table(df, model_names):
    """結果を整形し、色分け情報を含むテキストを出力"""

    model_1 = model_names[0]
    model_2 = model_names[1]

    header = f"{'Metric':<30} | {model_1 + ' (%)':>15} | {model_2 + ' (%)':>15} | {'Change (%)':>10} | {'Sign':>5}"
    output_lines = ["=" * len(header), header, "=" * len(header)]

    # データ行
    for index, row in df.iterrows():
        model_1_pct = f"{row[model_1] * 100:.2f}%"
        model_2_pct = f"{row[model_2] * 100:.2f}%"

        # 変化率と符号
        change_pct = row['Percent Change'] * 100
        sign = ' '

        if change_pct > 0.001:
            sign = '↑'
        elif change_pct < -0.001:
            sign = '↓'

        change_str = f"{change_pct:+.2f}%"

        line = f"{index:<30} | {model_1_pct:>15} | {model_2_pct:>15} | {change_str:>10} | {sign:>5}"
        output_lines.append(line)

    output_lines.append("=" * len(header))
    return "\n".join(output_lines)


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(COMPARISON_RESULT_ROOT, exist_ok=True)

    if args.model_list:
        # 1. モデル特定
        model_1 = args.model_list[0]
        model_2 = args.model_list[1]

        model_1_dir = os.path.join(EVALUATION_ROOT, model_1)
        model_2_dir = os.path.join(EVALUATION_ROOT, model_2)

        model_1_file = find_final_score_file(model_1_dir)
        model_2_file = find_final_score_file(model_2_dir)

        print("\n" + "#" * 50)
        print(f"## ⚙️ 比較対象: {model_1} vs {model_2}")
        print("#" * 50)
        if not model_1_file or not model_2_file:
            print(f"➡️ スキップ: 必要な評価ファイルが見つかりません。")
            if not model_1_file:
                print(f"   [{model_1}]: {model_1_dir} 内にファイルが見つかりません。")
            if not model_2_file:
                print(f"   [{model_2}]: {model_2_dir} 内にファイルが見つかりません。")
            sys.exit(0)

        # 2. データのロード
        data_model_1 = load_score(model_1_file)
        data_model_2 = load_score(model_2_file)

        if not data_model_1 or not data_model_2:
            print("➡️ スキップ: データのロードに失敗しました。")
            sys.exit(0)

        df = pd.DataFrame({
            model_1: data_model_1,
            model_2: data_model_2
        })

        # インデックス名の整形
        df.index = [format_index(i) for i in df.index]
        df = df.round(4)

        # 差分と変化率の計算
        df['Absolute Diff'] = df[model_2] - df[model_1]
        df['Percent Change'] = df['Absolute Diff'] / \
            df[model_1].replace(0, 1e-6)


        # 3. 結果出力・画像保存
        print(f"\n📈 {model_2} と {model_1} の比較結果:")
        model_names = [model_1, model_2]
        print(format_output_table(df, model_names))

        saved_path = save_table_as_image(df, model_1, model_2)
        print(f"\n🖼️ 比較結果の画像を保存しました: {saved_path}")

    else:
        for model_info in MODELS_TO_COMPARE:
            # 1. モデル特定
            base_name = str(model_info['base_model'])
            icl_num = str(model_info['icl_num'])

            base_dir = os.path.join(EVALUATION_ROOT, base_name)
            icl_name = f"{base_name}_ICL{icl_num}"
            icl_dir = os.path.join(EVALUATION_ROOT, icl_name)

            base_file = find_final_score_file(base_dir)
            icl_file = find_final_score_file(icl_dir)

            print("\n" + "#" * 50)
            print(f"## ⚙️ 比較対象: {base_name} (Base) vs {icl_name}")
            print("#" * 50)

            if not base_file or not icl_file:
                print(f"➡️ スキップ: 必要な評価ファイルが見つかりません。")
                if not base_file:
                    print(f"   [{base_name}]: {base_dir} 内にファイルが見つかりません。")
                if not icl_file:
                    print(f"   [{icl_name}]: {icl_dir} 内にファイルが見つかりません。")
                continue


            # 2. データのロード
            data_base = load_score(base_file)
            data_icl = load_score(icl_file)

            if not data_base or not data_icl:
                continue

            df = pd.DataFrame({
                base_name: data_base,
                icl_name: data_icl
            })

            # インデックス名の整形
            df.index = [format_index(i) for i in df.index]
            df = df.round(4)

            # 差分と変化率の計算
            df['Absolute Diff'] = df[icl_name] - df[base_name]
            df['Percent Change'] = df['Absolute Diff'] / \
                df[base_name].replace(0, 1e-6)


            # 3. 結果出力・画像保存
            print(f"\n📈 {icl_name} と {base_name} の比較結果:")
            model_names = [base_name, icl_name]
            print(format_output_table(df, model_names))

            saved_path = save_table_as_image(df, base_name, icl_name)
            print(f"\n🖼️ 比較結果の画像を保存しました: {saved_path}")
