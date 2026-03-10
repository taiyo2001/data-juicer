import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

import torch
import json
import tqdm
import statistics
from diffusers import DiffusionPipeline
from sd_embed.src.sd_embed.chunking import (
    get_prompts_tokens_with_weights,
    group_tokens_and_weights_clip,
    sentence_group_tokens_and_weights_clip,
)

prompt_path = "./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json"
output_path = "./data-juicer/outputs/DetailMaster_Dataset_chunking_comparison.json"
model_path = "black-forest-labs/FLUX.1-schnell"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
tokenizer = pipe.tokenizer  # CLIPTokenizer

EOS = 49407  # CLIP EOS token id


def count_content_tokens(chunk: list) -> int:
    """
    チャンク内の本文トークン数（BOS・EOSパディング・末尾EOS を除く）を返す。

    _add_chunk が生成するチャンク構造:
        [BOS, t1, ..., tn, EOS_pad*, EOS_final]
    chunk[1:-1] で BOS と末尾 EOS を除き、残りの EOS はパディングなので除外する。
    """
    content_region = chunk[1:-1]  # BOS と末尾 EOS を除く（最大 75 トークン）
    return sum(1 for t in content_region if t != EOS)


def analyze_chunks(chunks: list) -> dict:
    """チャンクリストの情報密度を集計する。"""
    n_chunks = len(chunks)
    content_per_chunk = [count_content_tokens(c) for c in chunks]
    total_content = sum(content_per_chunk)
    total_slots = n_chunks * 75          # 各チャンクの本文スロット容量
    total_padding = total_slots - total_content
    # チャンクあたりの平均本文トークン数（greedy ≈ 75、sentence < 75）
    avg_content_per_chunk = total_content / n_chunks if n_chunks > 0 else 0.0

    return {
        "n_chunks": n_chunks,
        "total_content_tokens": total_content,
        "total_slot_capacity": total_slots,
        "total_padding_tokens": total_padding,
        # チャンクあたり平均で何トークンの本文を運んでいるか（最大 75）
        "avg_content_per_chunk": round(avg_content_per_chunk, 2),
        "content_per_chunk": content_per_chunk,
    }


results = []

with open(prompt_path, "r") as f:
    data = json.load(f)

force_sentence_split = False
pad_last_block = False

for item in tqdm.tqdm(data):
    try:
        prompt = item["polished_prompt"]
        image_id = item["image_id"]

        # --- Method 1: greedy packing (group_tokens_and_weights_clip) ---
        token_ids, weights = get_prompts_tokens_with_weights(tokenizer, prompt)
        greedy_chunks, _ = group_tokens_and_weights_clip(token_ids, weights)
        greedy = analyze_chunks(greedy_chunks)

        # --- Method 2: sentence-aware packing (sentence_group_tokens_and_weights_clip) ---
        sent_chunks, _ = sentence_group_tokens_and_weights_clip(
            prompt,
            tokenizer,
            force_sentence_split=force_sentence_split,
            pad_last_block=pad_last_block,
        )
        sentence = analyze_chunks(sent_chunks)

        # --- Comparison ---
        # チャンクあたりの平均本文トークン数の低下量（正の値 = sentence の方が少ない）
        avg_token_drop = round(
            greedy["avg_content_per_chunk"] - sentence["avg_content_per_chunk"], 2
        )
        # greedy の avg_content_per_chunk に対する低下率 [%]
        avg_token_drop_pct = round(
            avg_token_drop / greedy["avg_content_per_chunk"] * 100, 2
        ) if greedy["avg_content_per_chunk"] > 0 else 0.0
        # sentence 法が生み出した余分なチャンク数
        chunk_overhead = sentence["n_chunks"] - greedy["n_chunks"]
        # sentence 法で増えたパディングトークン数（情報スロットの無駄）
        padding_overhead = sentence["total_padding_tokens"] - greedy["total_padding_tokens"]

        results.append({
            "image_id": image_id,
            "prompt_token_count": greedy["total_content_tokens"],
            "greedy": greedy,
            "sentence": sentence,
            "comparison": {
                # ★ チャンクあたりの平均本文トークン数の低下量（正 = sentence が少ない）
                #    greedy ≈ 75 に対して sentence がいくつ少ないか
                "avg_content_per_chunk_drop": avg_token_drop,
                # ★ 上記の greedy 比の低下率 [%]
                "avg_content_per_chunk_drop_pct": avg_token_drop_pct,
                # 正の値 = sentence 法が余分なチャンクを作っている
                "chunk_overhead": chunk_overhead,
                # 正の値 = sentence 法が無駄にしているパディングスロット数
                "padding_overhead_tokens": padding_overhead,
            },
        })

    except Exception as e:
        print(f"\n--- ERROR for {item.get('image_id')} ---")
        print(e)
        print("------------------------------------------")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# --- Summary statistics ---
if results:
    drops     = [r["comparison"]["avg_content_per_chunk_drop"] for r in results]
    drop_pcts = [r["comparison"]["avg_content_per_chunk_drop_pct"] for r in results]
    overheads = [r["comparison"]["padding_overhead_tokens"] for r in results]
    chunk_overheads = [r["comparison"]["chunk_overhead"] for r in results]
    greedy_avgs  = [r["greedy"]["avg_content_per_chunk"] for r in results]
    sentence_avgs = [r["sentence"]["avg_content_per_chunk"] for r in results]

    print("\n===== Chunking Comparison Summary =====")
    print(f"Samples analyzed : {len(results)}")
    print()
    print(f"--- avg_content_per_chunk (本文トークン数 / チャンク数、最大 75) ---")
    print(f"  greedy   mean  : {statistics.mean(greedy_avgs):.2f} tokens/chunk")
    print(f"  sentence mean  : {statistics.mean(sentence_avgs):.2f} tokens/chunk")
    print()
    print(f"--- avg_content_per_chunk_drop (greedy - sentence) [tokens/chunk] ---")
    print(f"  mean  : {statistics.mean(drops):.2f}")
    print(f"  median: {statistics.median(drops):.2f}")
    print(f"  min   : {min(drops):.2f}")
    print(f"  max   : {max(drops):.2f}")
    print()
    print(f"--- avg_content_per_chunk_drop_pct (drop / greedy_avg × 100) [%] ---")
    print(f"  mean  : {statistics.mean(drop_pcts):.2f}")
    print(f"  median: {statistics.median(drop_pcts):.2f}")
    print()
    print(f"--- padding_overhead_tokens (sentence - greedy の余分なパディング) ---")
    print(f"  mean  : {statistics.mean(overheads):.1f}")
    print(f"  median: {statistics.median(overheads):.1f}")
    print(f"  max   : {max(overheads)}")
    print()
    print(f"--- chunk_overhead (sentence が生み出した余分なチャンク数) ---")
    print(f"  mean  : {statistics.mean(chunk_overheads):.2f}")
    print(f"  median: {statistics.median(chunk_overheads):.1f}")
    print(f"\nResults saved to: {output_path}")
