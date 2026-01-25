import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- 設定 ---
output_path = "./data-juicer/outputs/DetailMaster_Dataset_analysis.json"
save_dir = "./data-juicer/outputs"
os.makedirs(save_dir, exist_ok=True)

with open(output_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- 1. ヒストグラムの描画 ---
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# 各トークナイザーの分布を重ねる
sns.histplot(df['clip_tokens'], kde=True, color="skyblue", label="CLIP (ViT-L/14)", bins=30, alpha=0.4)
sns.histplot(df['t5_tokens'], kde=True, color="salmon", label="T5-XXL", bins=30, alpha=0.4)
sns.histplot(df['qwen_vl_tokens'], kde=True, color="green", label="Qwen-VL (Original)", bins=30, alpha=0.4)
# sns.histplot(df['st1_qwen_vl_tokens'], kde=True, color="purple", label="Qwen-VL (old Structured)", bins=30, alpha=0.4)
sns.histplot(df['st3_qwen_vl_tokens'], kde=True, color="purple", label="Qwen-VL (Structured)", bins=30, alpha=0.4)
sns.histplot(df['dp_qwen_vl_tokens'], kde=True, color="orange", label="Qwen-VL (Dense)", bins=30, alpha=0.5)

# 制限ラインの追加
plt.axvline(x=77, color='blue', linestyle='--', alpha=0.5, label='CLIP Limit (77)')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.5, label='T5 Recommended (256)')


plt.title("Detailed Token Count Distribution", fontsize=15)
plt.xlabel("Number of Tokens", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()

plt.savefig(os.path.join(save_dir, "token_distribution_extended.png"))
plt.show()

# --- 2. 統計サマリーの表示 ---
print("\n--- Statistical Summary ---")
cols = ['clip_tokens', 't5_tokens', 'qwen_vl_tokens', 'st1_qwen_vl_tokens', 'st3_qwen_vl_tokens', 'dp_qwen_vl_tokens']
print(df[cols].describe())

# --- 3. トークン長の詳細分析 ---
total_count = len(df)

def print_threshold_stats(column_name, threshold):
    count = (df[column_name] > threshold).sum()
    percent = (count / total_count) * 100
    print(f"[{column_name}] Over {threshold} tokens: {count} samples ({percent:.2f}%)")

print(f"\n--- Token Length Analysis (Total: {total_count}) ---")
print(">> Threshold: 512 tokens")
print_threshold_stats('t5_tokens', 512)
print_threshold_stats('qwen_vl_tokens', 512)
print_threshold_stats('st1_qwen_vl_tokens', 512)
print_threshold_stats('st3_qwen_vl_tokens', 512)
print_threshold_stats('dp_qwen_vl_tokens', 512)

print("\n>> Threshold: 1024 tokens (Long Prompt Context)")
print_threshold_stats('t5_tokens', 1024)
print_threshold_stats('qwen_vl_tokens', 1024)
print_threshold_stats('st1_qwen_vl_tokens', 1024)
print_threshold_stats('st3_qwen_vl_tokens', 1024)
print_threshold_stats('dp_qwen_vl_tokens', 1024)

 # Structured Prompt
df['qwen_increase_st1'] = df['st1_qwen_vl_tokens'] - df['qwen_vl_tokens']
df['qwen_increase_st3'] = df['st3_qwen_vl_tokens'] - df['qwen_vl_tokens']
print(f"\n--- Structured Prompt Growth ---")
print(f"[Structured] Average: {df['st1_qwen_vl_tokens'].mean():.2f} tokens")
print(f"[Structured] Average token increase: {df['qwen_increase_st1'].mean():.2f}")
print(f"[Structured] Max token increase: {df['qwen_increase_st1'].max()}")

print(f"[Structured3] Average: {df['st3_qwen_vl_tokens'].mean():.2f} tokens")
print(f"[Structured3] Average token increase: {df['qwen_increase_st3'].mean():.2f}")
print(f"[Structured3] Max token increase: {df['qwen_increase_st3'].max()}")

# Dense Prompt
df['dp_change'] = df['dp_qwen_vl_tokens'] - df['qwen_vl_tokens']
print(f"\n--- Dense Prompt Change ---")
print(f"[Dense] Average: {df['dp_qwen_vl_tokens'].mean():.2f} tokens")
print(f"[Dense] Average change: {df['dp_change'].mean():.2f} tokens")
print(f"[Dense] Max reduction (saved tokens): {df['dp_change'].min() * -1 if df['dp_change'].min() < 0 else 0}")
