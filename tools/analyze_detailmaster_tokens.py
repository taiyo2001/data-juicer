import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

output_path = "./outputs/DetailMaster_Dataset_analysis.json"
with open(output_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")


sns.histplot(df['clip_tokens'], kde=True, color="skyblue", label="CLIP (ViT-L/14)", bins=30)
sns.histplot(df['t5_tokens'], kde=True, color="salmon", label="T5-XXL", bins=30)


plt.axvline(x=77, color='blue', linestyle='--', alpha=0.5, label='CLIP Limit (77)')
plt.axvline(x=256, color='red', linestyle='--', alpha=0.5, label='T5 Recommended (256)')


plt.title("Token Count Distribution: DetailMaster Dataset", fontsize=15)
plt.xlabel("Number of Tokens", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()

plt.savefig("./outputs/token_distribution.png")
plt.show()

print("--- Statistical Summary ---")
print(df[['clip_tokens', 't5_tokens']].describe())


total_count = len(df)

over_512_count = (df['t5_tokens'] > 512).sum()
over_512_percent = (over_512_count / total_count) * 100

over_768_count = (df['t5_tokens'] > 768).sum()
over_768_percent = (over_768_count / total_count) * 100

print(f"--- Token Length Analysis ---")
print(f"Total samples: {total_count}")
print(f"Over 512 tokens: {over_512_count} samples ({over_512_percent:.2f}%)")
print(f"Over 768 tokens: {over_768_count} samples ({over_768_percent:.2f}%)")
