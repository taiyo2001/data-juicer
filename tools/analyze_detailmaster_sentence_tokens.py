import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import japanize_matplotlib

input_path = "./data-juicer/outputs/DetailMaster_Dataset_sentence_analysis.json"
save_dir = "./data-juicer/outputs"
os.makedirs(save_dir, exist_ok=True)

with open(input_path, "r") as f:
    data = json.load(f)

sentence_list = []
for item in data:
    for sent in item.get("sentences", []):
        sentence_list.append({
            "image_id": item["image_id"],
            "clip_token_count": sent["clip_token_count"]
        })

df_sent = pd.DataFrame(sentence_list)

# --- 1. ヒストグラムと統計情報の描画 ---
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")
japanize_matplotlib.japanize()

# CLIPトークン数の分布を描画
sns.histplot(df_sent['clip_token_count'], kde=True, color="teal", label="CLIP Token count (per sentence)", bins=40, alpha=0.6)

mean_val = df_sent['clip_token_count'].mean()
max_val = df_sent['clip_token_count'].max()
min_val = df_sent['clip_token_count'].min()
mode_val = df_sent['clip_token_count'].mode()[0]
median_val = df_sent['clip_token_count'].median()

plt.axvline(x=77, color='red', linestyle='--', linewidth=2, label='CLIP Block Limit (77)')
# plt.axvline(x=mean_val, color='orange', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
# plt.axvline(x=mode_val, color='purple', linestyle=':', linewidth=1.5, label=f'Mode: {mode_val}')

plt.title("Analysis of Token Count Distribution per Sentence in Prompts", fontsize=18)
plt.xlabel("Number of Tokens per Sentence", fontsize=14)
plt.ylabel("Frequency (Number of Sentences)", fontsize=14)
plt.legend(fontsize=12)

plt.savefig(os.path.join(save_dir, "sentence_token_distribution.png"), dpi=300)
plt.show()

# --- 2. 統計サマリーの表示 ---
print("\n" + "="*50)
print("   SENTENCE-LEVEL TOKEN STATISTICS SUMMARY")
print("="*50)
print(f"総分析文数    : {len(df_sent)} 文")
print(f"平均トークン数: {mean_val:.2f}")
print(f"中央値        : {median_val:.2f}")
print(f"最頻値        : {mode_val}")
print(f"最大値        : {max_val}")
print(f"最小値        : {min_val}")
print(f"標準偏差      : {df_sent['clip_token_count'].std():.2f}")
print("-"*50)

# --- 3. 閾値分析 (CLIP制限 77トークンとの比較) ---
threshold = 77
over_threshold = df_sent[df_sent['clip_token_count'] > threshold]
count_over = len(over_threshold)
percent_over = (count_over / len(df_sent)) * 100

print(f"\n--- CLIP Block Limit Analysis (Threshold: {threshold}) ---")
print(f"1ブロック(77)を超える文の数: {count_over} 文")
print(f"全体に対する割合           : {percent_over:.2f} %")

if count_over > 0:
    print("\n>> 非常に長い文を持つ Image IDs (Top 5):")
    print(over_threshold.sort_values(by='clip_token_count', ascending=False)[['image_id', 'clip_token_count']].head(5))

print("\n" + "="*50)
