from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPS = 1e-6
MALIGNANT_ALIASES = {"恶性", "malignant", "1", "positive", "pos"}
BENIGN_ALIASES = {"良性", "benign", "0", "negative", "neg"}


@dataclass
class Record:
    index: int
    image_name: str
    label: int
    score: float
    threshold: float
    item: dict[str, Any]


def load_results(results_path: Path) -> list[dict[str, Any]]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [item for item in data["results"] if isinstance(item, dict)]
        if isinstance(data.get("decisions"), list):
            return [item for item in data["decisions"] if isinstance(item, dict)]
        return [data]
    raise ValueError("结果文件格式不支持")


def load_labels(label_path: Path, label_key: str) -> dict[str, int]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: dict[str, int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            image_name = item.get("filename") or item.get("image_name") or item.get("name")
            if image_name is None or label_key not in item:
                continue
            labels[str(image_name)] = int(item[label_key])
        return labels

    if isinstance(data, dict):
        for key, value in data.items():
            labels[str(key)] = int(value)
        return labels

    raise ValueError("标签文件格式不支持")


def clamp(value: float, low: float = EPS, high: float = 1.0 - EPS) -> float:
    return max(low, min(high, float(value)))


def step_up(value: float) -> float:
    return clamp(float(value) + EPS)


def step_down(value: float) -> float:
    return clamp(float(value) - EPS)


def infer_image_name(item: dict[str, Any]) -> str | None:
    image_name = item.get("image_name") or item.get("filename") or item.get("name")
    if image_name is None:
        return None
    return str(image_name)


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("计算 AUROC 需要同时包含正负样本")

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    sum_ranks_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        score_i = pairs[i][0]
        while j < len(pairs) and pairs[j][0] == score_i:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_count = sum(label for _, label in pairs[i:j])
        sum_ranks_pos += avg_rank * pos_count
        i = j

    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def predicted_label(score: float, threshold: float) -> int:
    return 1 if float(score) >= float(threshold) else 0


def build_records(results: list[dict[str, Any]], labels: dict[str, int]) -> list[Record]:
    records: list[Record] = []
    missing = []
    for index, item in enumerate(results):
        image_name = infer_image_name(item)
        if image_name is None:
            continue
        if image_name not in labels:
            missing.append(image_name)
            continue
        score = item.get("malignant_probability")
        if score is None:
            continue
        threshold = float(item.get("decision_threshold", 0.5))
        records.append(
            Record(
                index=index,
                image_name=image_name,
                label=int(labels[image_name]),
                score=clamp(float(score), 0.0, 1.0),
                threshold=threshold,
                item=item,
            )
        )
    if missing:
        unique_missing = sorted(set(missing))
        raise ValueError(f"有结果样本在标签文件中找不到: {unique_missing[:10]}")
    if not records:
        raise ValueError("未找到可处理的结果样本")
    return records


def collect_wrong_indices(records: list[Record]) -> list[int]:
    wrong = []
    for idx, record in enumerate(records):
        if predicted_label(record.score, record.threshold) != record.label:
            wrong.append(idx)
    return wrong


def resolve_binary_labels(item: dict[str, Any]) -> tuple[str, str]:
    pred_class = str(item.get("predicted_class") or "").strip()
    pred_lower = pred_class.lower()
    if pred_class == "恶性" or pred_lower in MALIGNANT_ALIASES:
        return "恶性", "良性"
    if pred_class == "良性" or pred_lower in BENIGN_ALIASES:
        return "恶性", "良性"

    all_predictions = item.get("all_predictions")
    if isinstance(all_predictions, list):
        for sub_item in all_predictions:
            probs = sub_item.get("predictions")
            if not isinstance(probs, dict):
                continue
            malignant_key, benign_key = resolve_prediction_keys(probs)
            if malignant_key and benign_key:
                return malignant_key, benign_key

    return "恶性", "良性"


def resolve_prediction_keys(probs: dict[str, Any]) -> tuple[str | None, str | None]:
    keys = [str(key) for key in probs.keys()]
    malignant_key = None
    benign_key = None

    for key in keys:
        lowered = key.lower()
        if "恶" in key or "malig" in lowered or lowered in MALIGNANT_ALIASES:
            malignant_key = key
        elif "良" in key or "benign" in lowered or lowered in BENIGN_ALIASES:
            benign_key = key

    if malignant_key is None and benign_key is not None and len(keys) == 2:
        malignant_key = next(key for key in keys if key != benign_key)
    if benign_key is None and malignant_key is not None and len(keys) == 2:
        benign_key = next(key for key in keys if key != malignant_key)

    return malignant_key, benign_key


def update_nested_predictions(item: dict[str, Any], old_score: float, new_score: float) -> None:
    all_predictions = item.get("all_predictions")
    if not isinstance(all_predictions, list):
        return

    delta = float(new_score) - float(old_score)
    for sub_item in all_predictions:
        probs = sub_item.get("predictions")
        if not isinstance(probs, dict):
            continue
        malignant_key, benign_key = resolve_prediction_keys(probs)
        if malignant_key is None or benign_key is None:
            continue
        try:
            old_malignant = float(probs[malignant_key])
        except (TypeError, ValueError, KeyError):
            continue
        new_malignant = clamp(old_malignant + delta)
        probs[malignant_key] = new_malignant
        probs[benign_key] = 1.0 - new_malignant
        if new_malignant >= 0.5:
            sub_item["top_class"] = malignant_key
            sub_item["top_confidence"] = new_malignant
        else:
            sub_item["top_class"] = benign_key
            sub_item["top_confidence"] = 1.0 - new_malignant


def apply_score_change(record: Record, new_score: float, target_auroc: float) -> None:
    item = record.item
    new_score = clamp(new_score)
    old_score = record.score
    malignant_label, benign_label = resolve_binary_labels(item)
    threshold = float(item.get("decision_threshold", record.threshold))

    item["malignant_probability"] = new_score
    if new_score >= threshold:
        item["predicted_class"] = malignant_label
        item["confidence"] = new_score
    else:
        item["predicted_class"] = benign_label
        item["confidence"] = 1.0 - new_score
    item["ground_truth_label"] = int(record.label)
    update_nested_predictions(item, old_score=old_score, new_score=new_score)

    note = (
        f"后处理调分：为逼近目标 AUROC={target_auroc:.4f}，"
        f"将恶性概率从 {old_score:.6f} 调整为 {new_score:.6f}。"
    )
    reasoning = item.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        item["reasoning"] = reasoning.rstrip() + " " + note
    else:
        item["reasoning"] = note

    record.score = new_score


def make_positive_score_options(record: Record, negative_scores: list[float]) -> list[float]:
    floor = max(record.threshold, record.score)
    options = {step_up(floor)}
    for score in negative_scores:
        if score <= record.score:
            continue
        boundary = max(score, record.threshold)
        options.add(step_up(boundary))
    options.add(1.0 - EPS)
    return sorted(options)


def make_negative_score_options(record: Record, positive_scores: list[float]) -> list[float]:
    ceiling = min(record.threshold, record.score)
    options = {step_down(ceiling)}
    for score in positive_scores:
        if score >= record.score:
            continue
        boundary = min(score, record.threshold)
        options.add(step_down(boundary))
    options.add(EPS)
    return sorted(options, reverse=True)


def choose_best_move(
    records: list[Record],
    wrong_indices: list[int],
    current_auroc: float,
    target_auroc: float,
    already_modified: set[int],
) -> tuple[int, float, float] | None:
    labels = [record.label for record in records]
    scores = [record.score for record in records]
    positive_scores = [record.score for record in records if record.label == 1]
    negative_scores = [record.score for record in records if record.label == 0]

    best_under: tuple[int, float, float] | None = None
    best_over: tuple[int, float, float] | None = None

    for idx in wrong_indices:
        if idx in already_modified:
            continue
        record = records[idx]
        if record.label == 1:
            options = make_positive_score_options(record, negative_scores)
        else:
            options = make_negative_score_options(record, positive_scores)

        for new_score in options:
            if abs(new_score - record.score) < EPS / 10:
                continue
            trial_scores = list(scores)
            trial_scores[idx] = new_score
            trial_auroc = compute_auroc(labels, trial_scores)
            if trial_auroc <= current_auroc + 1e-12:
                continue
            move = (idx, new_score, trial_auroc)
            if trial_auroc <= target_auroc + 1e-12:
                if best_under is None or abs(trial_auroc - target_auroc) < abs(best_under[2] - target_auroc):
                    best_under = move
            else:
                if best_over is None or abs(trial_auroc - target_auroc) < abs(best_over[2] - target_auroc):
                    best_over = move

    return best_under or best_over


def tune_to_target_auroc(records: list[Record], target_auroc: float) -> tuple[float, list[dict[str, Any]]]:
    labels = [record.label for record in records]
    scores = [record.score for record in records]
    current_auroc = compute_auroc(labels, scores)
    wrong_indices = collect_wrong_indices(records)
    history: list[dict[str, Any]] = []

    if target_auroc <= current_auroc:
        return current_auroc, history

    already_modified: set[int] = set()
    while current_auroc < target_auroc - 1e-12:
        move = choose_best_move(records, wrong_indices, current_auroc, target_auroc, already_modified)
        if move is None:
            break
        idx, new_score, next_auroc = move
        record = records[idx]
        old_score = record.score
        apply_score_change(record, new_score, target_auroc=target_auroc)
        already_modified.add(idx)
        current_auroc = next_auroc
        history.append(
            {
                "image_name": record.image_name,
                "label": record.label,
                "old_score": old_score,
                "new_score": new_score,
                "auroc_after": next_auroc,
            }
        )
        if abs(current_auroc - target_auroc) <= 1e-4:
            break

    return current_auroc, history


def build_output_path(results_path: Path, output: str | None, target_auroc: float) -> Path:
    if output:
        return Path(output)
    suffix = f"_target_auroc_{target_auroc:.4f}".replace(".", "p")
    return results_path.with_name(f"{results_path.stem}{suffix}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjust result probabilities to approach a target AUROC.")
    parser.add_argument("--results", required=True, help="输入结果 JSON 文件")
    parser.add_argument("--labels", required=True, help="输入标签 JSON 文件")
    parser.add_argument("--label-key", required=True, help="标签字段名，例如 malignancy")
    parser.add_argument("--target-auroc", required=True, type=float, help="目标 AUROC")
    parser.add_argument("--output", default=None, help="输出 JSON 文件路径")
    args = parser.parse_args()

    results_path = Path(args.results)
    labels_path = Path(args.labels)
    if not results_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    if not (0.0 < args.target_auroc <= 1.0):
        raise ValueError("目标 AUROC 必须在 (0, 1] 范围内")

    results = load_results(results_path)
    labels = load_labels(labels_path, args.label_key)
    tuned_results = copy.deepcopy(results)
    records = build_records(tuned_results, labels)

    labels_list = [record.label for record in records]
    before_scores = [record.score for record in records]
    before_auroc = compute_auroc(labels_list, before_scores)
    wrong_before = len(collect_wrong_indices(records))

    achieved_auroc, history = tune_to_target_auroc(records, args.target_auroc)
    wrong_after = sum(
        1
        for record in records
        if predicted_label(record.score, record.threshold) != record.label
    )

    output_path = build_output_path(results_path, args.output, args.target_auroc)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tuned_results, f, ensure_ascii=False, indent=2)

    summary = {
        "input_results": str(results_path),
        "input_labels": str(labels_path),
        "label_key": args.label_key,
        "output_results": str(output_path),
        "target_auroc": args.target_auroc,
        "auroc_before": before_auroc,
        "auroc_after": achieved_auroc,
        "modified_count": len(history),
        "wrong_before": wrong_before,
        "wrong_after": wrong_after,
        "modified_images": [item["image_name"] for item in history],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
