import os
import sys

# --- Dynamic Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"Project Root Directory: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)
# --------------------------------

from google import genai
import statistics
import argparse
import tqdm
import time
import json
from src.constants import DETAIL_MASTER

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="gemini-3-flash-preview")
    parser.add_argument('--ann_json_path', type=str,
                        default="./data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json")
    parser.add_argument("--aggregated_prompt_json", type=str,
                        default="./data-juicer/outputs/llm_prompt/output_dp1_info_gemini-3-flash-preview.json")
    parser.add_argument("--output_json", type=str,
                        default="./data-juicer/outputs/analysis/prompt_retention_eval.json")
    args = parser.parse_args()

    return args


# ---------------------------------------------------------------------------
# Criteria builders
# ---------------------------------------------------------------------------

def build_criteria(ann_item: dict) -> list[dict]:
    """
    Build a flat list of evaluation criteria from one annotation item.

    Follows the same category structure as eval_process.py:
      1. object_presence    – is the character mentioned at all?
      2. character_attributes – are specific attributes described?
      3. character_locations  – is the spatial position mentioned?
      4. scene_attributes     – background / light / style / spatial
    """
    criteria = []

    # --- object_presence --------------------------------------------------
    # Union of main_characters from character_attributes + character_locations
    # (same logic as valid_character_list in eval_process.py)
    seen_chars = set()
    for char_attr in ann_item.get("character_attributes", []):
        mc = char_attr["main_character"]
        if mc not in seen_chars:
            seen_chars.add(mc)
            criteria.append({
                "category": "object_presence",
                "main_character": mc,
                "question": (
                    f'Does the aggregated prompt mention or reference "{mc}"?'
                ),
            })
    for loc in ann_item.get("character_locations", []):
        mc = loc["main_character"]
        if mc not in seen_chars:
            seen_chars.add(mc)
            criteria.append({
                "category": "object_presence",
                "main_character": mc,
                "question": (
                    f'Does the aggregated prompt mention or reference "{mc}"?'
                ),
            })

    # --- character_attributes ---------------------------------------------
    for char_attr in ann_item.get("character_attributes", []):
        mc = char_attr["main_character"]
        cls = char_attr.get("cls", "")
        for attr in char_attr.get("characteristics_list", []):
            criteria.append({
                "category": "character_attributes",
                "main_character": mc,
                "cls": cls,
                "attribute": attr,
                "question": (
                    f'Does the aggregated prompt describe "{mc}" as having '
                    f'the characteristic "{attr}"?'
                ),
            })

    # --- character_locations ----------------------------------------------
    for loc in ann_item.get("character_locations", []):
        mc = loc["main_character"]
        pos = loc["position"]
        criteria.append({
            "category": "character_locations",
            "main_character": mc,
            "position": pos,
            "question": (
                f'Does the aggregated prompt indicate that "{mc}" is located '
                f'in "{pos}"?'
            ),
        })

    # --- scene_attributes -------------------------------------------------
    for scene_attr in ann_item.get("scene_attributes", []):
        attr_type = scene_attr["scene_attribute"]
        content = scene_attr["content"]
        if attr_type != "spatial":
            criteria.append({
                "category": "scene_attributes",
                "scene_attribute": attr_type,
                "content": content,
                "question": (
                    f'Does the aggregated prompt convey the following about '
                    f'{attr_type}: "{content}"?'
                ),
            })
        else:
            # spatial content is a list of relationship strings
            for spatial_rel in content:
                criteria.append({
                    "category": "scene_attributes",
                    "scene_attribute": "spatial",
                    "content": spatial_rel,
                    "question": (
                        f'Does the aggregated prompt convey the spatial '
                        f'relationship: "{spatial_rel}"?'
                    ),
                })

    return criteria


# ---------------------------------------------------------------------------
# LLM evaluation prompt
# ---------------------------------------------------------------------------

def build_eval_prompt(polished_prompt: str, aggregated_prompt: str, criteria: list[dict]) -> str:
    """
    Build a single batched evaluation prompt.
    All criteria for one item are evaluated in one API call to reduce latency.
    """
    criteria_lines = [f'[{i+1}] {c["question"]}' for i,
                      c in enumerate(criteria)]
    criteria_text = "\n".join(criteria_lines)

    return f"""You are evaluating whether an aggregated prompt retains specific information from the original prompt.

Original prompt:
{polished_prompt}

Aggregated prompt:
{aggregated_prompt}

For each numbered criterion below, answer 'yes' if the aggregated prompt explicitly or implicitly contains the information, or 'no' if the information is absent or unclear.
Respond with ONLY a JSON object mapping each number (as a string key) to "yes" or "no".
Example format: {{"1": "yes", "2": "no", "3": "yes"}}

Criteria:
{criteria_text}"""


def parse_llm_response(response_text: str, n_criteria: int) -> list[bool] | None:
    """Parse LLM JSON response into a bool list (True = retained).

    Returns None if parsing fails.
    """
    text = response_text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    try:
        result_dict = json.loads(text)
        return [result_dict.get(str(i), "no").strip().lower() == "yes" for i in range(1, n_criteria + 1)]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_category_stats(eval_results: list[dict]) -> dict:
    """Aggregate per-category retention counts across all evaluated items."""
    cats = {
        "object_presence":      {"all": 0, "success": 0},
        "character_attributes": {"all": 0, "success": 0},
        "character_locations":  {"all": 0, "success": 0},
        "scene_attributes":     {"all": 0, "success": 0, "by_type": {}},
    }

    for result in eval_results:
        for c in result.get("criteria_results", []):
            cat = c["category"]
            retained = c["retained"]

            if cat in ("object_presence", "character_attributes", "character_locations"):
                cats[cat]["all"] += 1
                if retained:
                    cats[cat]["success"] += 1

            elif cat == "scene_attributes":
                cats["scene_attributes"]["all"] += 1
                if retained:
                    cats["scene_attributes"]["success"] += 1
                attr_type = c.get("scene_attribute", "unknown")
                by_type = cats["scene_attributes"]["by_type"]
                if attr_type not in by_type:
                    by_type[attr_type] = {"all": 0, "success": 0}
                by_type[attr_type]["all"] += 1
                if retained:
                    by_type[attr_type]["success"] += 1

    def pct(d):
        return round(d["success"] / d["all"] * 100, 2) if d["all"] > 0 else 0.0

    return {
        "object_presence": {
            "retention_pct": pct(cats["object_presence"]),
            **cats["object_presence"],
        },
        "character_attributes": {
            "retention_pct": pct(cats["character_attributes"]),
            **cats["character_attributes"],
        },
        "character_locations": {
            "retention_pct": pct(cats["character_locations"]),
            **cats["character_locations"],
        },
        "scene_attributes": {
            "retention_pct": pct(cats["scene_attributes"]),
            "all": cats["scene_attributes"]["all"],
            "success": cats["scene_attributes"]["success"],
            "by_type": {
                k: {"retention_pct": pct(v), **v}
                for k, v in cats["scene_attributes"]["by_type"].items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    client = genai.Client()

    # Load annotation data
    with open(args.ann_json_path, "r") as f:
        ann_data = json.load(f)

    # Load aggregated prompts, indexed by composite image_id
    with open(args.aggregated_prompt_json, "r") as f:
        agg_data = json.load(f)
    agg_dict = {item["image_id"]: item["llm_output"] for item in agg_data}

    # Checkpoint support
    eval_results = []
    processed_ids: set[str] = set()
    checkpoint_file = args.output_json.replace(".json", "_checkpoint.json")

    if os.path.exists(checkpoint_file):
        print(f"--- Checkpoint found! Resuming from {checkpoint_file} ---")
        with open(checkpoint_file, "r") as f:
            eval_results = json.load(f)
        processed_ids = {r["image_id"] for r in eval_results}
        print(f"--- Resuming from {len(processed_ids)} processed items ---")

    total_count = len(ann_data)
    report_step = max(1, total_count // 10)
    valid_count = len(processed_ids)
    no_agg_prompt_ids = []
    parse_failed_ids = []

    for ann_item in tqdm.tqdm(ann_data):
        # Composite image_id: same format as eval_process.py
        image_id = f"{ann_item['dataset_target']}_{ann_item['image_id']}"

        if image_id in processed_ids:
            continue

        agg_prompt = agg_dict.get(image_id)
        if not agg_prompt:
            no_agg_prompt_ids.append(image_id)
            continue

        polished_prompt = ann_item["polished_prompt"]
        criteria = build_criteria(ann_item)

        if not criteria:
            continue

        eval_prompt = build_eval_prompt(polished_prompt, agg_prompt, criteria)

        max_retries = DETAIL_MASTER.RETRY.MAX_RETRIES
        retry_count = 0
        success = False

        while retry_count <= max_retries and not success:
            try:
                response = client.models.generate_content(
                    model=args.model_name,
                    contents=eval_prompt,
                )
                retained_flags = parse_llm_response(
                    response.text, len(criteria))

                if retained_flags is None:
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(
                            f"\n--- Parse failed for {image_id}. Retrying {retry_count}/{max_retries} ---")
                        print(f"    raw response: {response.text[:200]}")
                        continue
                    # 全リトライ消費後もパース失敗
                    print(
                        f"\n--- Parse failed for {image_id} after {max_retries} retries. Skipping. ---")
                    parse_failed_ids.append(image_id)
                    valid_count += 1
                    break

                criteria_results = [
                    {**c, "retained": retained}
                    for c, retained in zip(criteria, retained_flags)
                ]

                n_total = len(criteria_results)
                n_success = sum(1 for c in criteria_results if c["retained"])

                eval_results.append({
                    "image_id": image_id,
                    "n_criteria": n_total,
                    "n_retained": n_success,
                    "retention_pct": round(n_success / n_total * 100, 2) if n_total > 0 else 0.0,
                    "criteria_results": criteria_results,
                    "raw_llm_response": response.text,
                })

                valid_count += 1
                success = True

                if valid_count % report_step == 0:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(eval_results, f,
                                  ensure_ascii=False, indent=2)
                    print(f"\nCheckpoint saved ({valid_count}/{total_count})")

            except Exception as e:
                error_str = str(e)
                if any(code in error_str for code in ("503", "429", "overloaded")):
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = retry_count * 5
                        print(
                            f"\n--- Rate limited. Retrying {retry_count}/{max_retries} after {wait_time}s ---")
                        time.sleep(wait_time)
                        continue

                print(f"\n--- ERROR for {image_id}: {e} ---")
                valid_count += 1
                break

    # Save full results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    # Compute and print summary
    category_stats = compute_category_stats(eval_results)
    per_item_rates = [r["retention_pct"] for r in eval_results]

    print("\n===== Prompt Retention Evaluation Summary =====")
    print(f"Samples evaluated            : {len(eval_results)}")
    print(f"No aggregated prompt found   : {len(no_agg_prompt_ids)}")
    print(f"LLM parse failed (skipped)   : {len(parse_failed_ids)}")
    print()

    if per_item_rates:
        print("--- Per-item retention rate [%] ---")
        print(f"  mean  : {statistics.mean(per_item_rates):.2f}")
        print(f"  median: {statistics.median(per_item_rates):.2f}")
        print(f"  min   : {min(per_item_rates):.2f}")
        print(f"  max   : {max(per_item_rates):.2f}")
        print()

    print("--- Category-wise retention [%] ---")
    op = category_stats["object_presence"]
    ca = category_stats["character_attributes"]
    cl = category_stats["character_locations"]
    sa = category_stats["scene_attributes"]

    print(
        f"  object_presence          : {op['retention_pct']:.2f}%  ({op['success']}/{op['all']})")
    print(
        f"  character_attributes     : {ca['retention_pct']:.2f}%  ({ca['success']}/{ca['all']})")
    print(
        f"  character_locations      : {cl['retention_pct']:.2f}%  ({cl['success']}/{cl['all']})")
    print(
        f"  scene_attributes (total) : {sa['retention_pct']:.2f}%  ({sa['success']}/{sa['all']})")
    for attr_type, v in sa["by_type"].items():
        print(
            f"    {attr_type:10s}          : {v['retention_pct']:.2f}%  ({v['success']}/{v['all']})")

    # Save summary JSON
    summary_path = args.output_json.replace(".json", "_summary.json")
    summary = {
        "aggregated_prompt_json": args.aggregated_prompt_json,
        "n_evaluated": len(eval_results),
        "n_no_agg_prompt": len(no_agg_prompt_ids),
        "n_parse_failed": len(parse_failed_ids),
        "parse_failed_ids": parse_failed_ids,
        "per_item_retention_pct": {
            "mean":   round(statistics.mean(per_item_rates), 2) if per_item_rates else 0.0,
            "median": round(statistics.median(per_item_rates), 2) if per_item_rates else 0.0,
            "min":    round(min(per_item_rates), 2) if per_item_rates else 0.0,
            "max":    round(max(per_item_rates), 2) if per_item_rates else 0.0,
        },
        "category_retention": category_stats,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to : {args.output_json}")
    print(f"Summary saved to : {summary_path}")
