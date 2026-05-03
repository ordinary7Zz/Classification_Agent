from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> list[dict]:
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
    return []


def extract_roc_inputs(results: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    y_true = []
    y_prob = []
    skipped = 0

    for item in results:
        if item.get("record_type") == "roc_summary":
            continue

        gt = item.get("true_label")
        prob = item.get("prob_class_1")
        if gt is None or prob is None:
            gt = item.get("ground_truth_label")
            prob = item.get("malignant_probability")

        if gt is None or prob is None:
            skipped += 1
            continue

        try:
            gt_int = int(gt)
            prob_float = float(prob)
        except (TypeError, ValueError):
            skipped += 1
            continue

        if gt_int not in {0, 1}:
            skipped += 1
            continue
        if not (0.0 <= prob_float <= 1.0):
            skipped += 1
            continue

        y_true.append(gt_int)
        y_prob.append(prob_float)

    return (
        np.asarray(y_true, dtype=np.int32),
        np.asarray(y_prob, dtype=np.float64),
        skipped,
    )


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str):
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少 scikit-learn，请先安装后再绘制 AUROC，例如: pip install scikit-learn"
        ) from e

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = float(roc_auc_score(y_true, y_prob))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color="gray", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return auroc


def main():
    parser = argparse.ArgumentParser(description="Plot AUROC directly from results_*.json")
    parser.add_argument("--results", required=True, help="Path to results_*.json")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--title", default="ROC Curve", help="Plot title")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_path}")

    if args.output is None:
        output_path = results_path.with_name(f"{results_path.stem}_auroc.png")
    else:
        output_path = Path(args.output)

    results = load_results(results_path)
    if not results:
        raise ValueError("结果文件中没有可读取的样本记录")

    y_true, y_prob, skipped = extract_roc_inputs(results)
    if y_true.size == 0:
        raise ValueError(
            "results_*.json 中没有可用于 AUROC 的样本。请确认文件包含 true_label/prob_class_1 或旧版 ground_truth_label/malignant_probability 字段。"
        )

    unique = np.unique(y_true)
    if unique.size < 2:
        raise ValueError("可用样本只包含单一类别，无法计算 AUROC 曲线")

    auroc = plot_roc(y_true, y_prob, output_path, args.title)

    print(f"结果文件: {results_path}")
    print(f"可用样本数: {len(y_true)}")
    print(f"跳过样本数: {skipped}")
    print(f"阳性样本数: {int(np.sum(y_true == 1))}")
    print(f"阴性样本数: {int(np.sum(y_true == 0))}")
    print(f"AUROC: {auroc:.4f}")
    print(f"ROC 图已保存: {output_path}")


if __name__ == "__main__":
    main()
