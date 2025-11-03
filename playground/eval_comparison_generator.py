import os
import sys
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
EVALUATION_ROOT = os.path.join(
    PROJECT_ROOT, "data-juicer/playground/evaluation/")
# --------------------------------

MODELS_TO_COMPARE = [
    {"base_model": "SD1_5", "icl_num": 1},
    {"base_model": "FLUX1-schnell", "icl_num": 1},
    {"base_model": "ParaDiffusion", "icl_num": 1},
    {"base_model": "ParaDiffusion", "icl_num": 2},
    # {"base_model": "ParaDiffusion_L", "icl_num": 1},
]


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


def save_table_as_image(df, filepath, title):
    """DataFrameの画像保存"""

    png_filepath = filepath.replace(".csv", ".png")

    df_display = df.copy()

    def format_pct_diff(row):
            pct_change = row['Percent Change'] * 100
            sign = '▲' if pct_change > 0.001 else ('▼' if pct_change < -0.001 else '—')
            return f"{pct_change:+.2f}% ({sign})"

    change_values = df_display['Percent Change']

    df_display['Base Score'] = (df_display['Base'] * 100).round(2).astype(str) + '%'
    df_display['ICL Score'] = (df_display['ICL'] * 100).round(2).astype(str) + '%'
    df_display['Change'] = df_display.apply(format_pct_diff, axis=1)

    df_final = df_display[['Base Score', 'ICL Score', 'Change']]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df_final.values,
                     colLabels=df_final.columns,
                     rowLabels=df_final.index,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)


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


    plt.title(title, y=1.05)

    plt.savefig(png_filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return png_filepath


def format_output_table(df, model_names):
    """結果を整形し、色分け情報を含むテキストを出力"""

    base_col = model_names['Base']
    icl_col = model_names['ICL']

    header = f"{'Metric':<30} | {base_col + ' (%)':>15} | {icl_col + ' (%)':>15} | {'Change (%)':>10} | {'Sign':>5}"
    output_lines = ["=" * len(header), header, "=" * len(header)]

    # データ行
    for index, row in df.iterrows():
        base_pct = f"{row['Base'] * 100:.2f}%"
        icl1_pct = f"{row['ICL'] * 100:.2f}%"

        # 変化率と符号
        change_pct = row['Percent Change'] * 100
        sign = ' '

        if change_pct > 0.001:
            sign = '↑'
        elif change_pct < -0.001:
            sign = '↓'

        change_str = f"{change_pct:+.2f}%"

        line = f"{index:<30} | {base_pct:>15} | {icl1_pct:>15} | {change_str:>10} | {sign:>5}"
        output_lines.append(line)

    output_lines.append("=" * len(header))
    return "\n".join(output_lines)


if __name__ == "__main__":

    for model_info in MODELS_TO_COMPARE:
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
                print(f"   [Base]: {base_dir} 内にファイルが見つかりません。")
            if not icl_file:
                print(f"   [ICL]: {icl_dir} 内にファイルが見つかりません。")
            continue


        # 2. データのロード
        data_base = load_score(base_file)
        data_icl = load_score(icl_file)

        if not data_base or not data_icl:
            continue

        df = pd.DataFrame({
            'Base': data_base,
            'ICL': data_icl
        })

        # インデックス名の整形
        df.index = [format_index(i) for i in df.index]
        df = df.round(4)

        # 差分と変化率の計算
        df['Absolute Diff'] = df['ICL'] - df['Base']
        df['Percent Change'] = df['Absolute Diff'] / \
            df['Base'].replace(0, 1e-6)


        # 3. 結果出力
        print(f"\n📈 {icl_name} と {base_name} の比較結果:")
        model_names = {'Base': base_name, 'ICL': icl_name}
        print(format_output_table(df, model_names))


        # 4. 画像保存
        image_output_name = f"{icl_name}_comparison_summary.png"
        image_filepath = os.path.join(EVALUATION_ROOT, image_output_name)

        table_title = f"Evaluation Comparison: {icl_name} vs {base_name}"

        saved_path = save_table_as_image(df, image_filepath, table_title)

        print(f"\n💾 画像保存完了: {saved_path}")
        print("-" * 50)
