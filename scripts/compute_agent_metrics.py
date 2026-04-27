"""
根据 Agent 推理输出结果和标签文件，计算分类指标及 CI95 置信区间。

指标: AUROC, AUPRC, Acc, Prec, Recall, F1, Specificity
"""

import json
from pathlib import Path

import numpy as np

# ============ 配置（直接修改此处） ============
CONFIG = {
    "results_path": "output/ThyroidXL/results_20260112_212017.json",  # Agent 推理结果 JSON
    "label_path": "ThyroidXL_test_label.json",                        # 标签 JSON
    "output_path": "output/ThyroidXL/agent_metrics.json",              # 输出 JSON（None 则不保存）
    "threshold": 0.5,
    "n_boot": 2000,
    "ci": 0.95,
    "seed": 0,
}
# ============================================
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def classification_bootstrap_metrics(
    y_probs, y_labels, threshold=0.5, n_boot=2000, ci=0.95, seed=0
):
    """
    使用 bootstrap 在样本级别估计二分类指标的均值及 CI95。
    返回 dict: 每个指标 (mean, (lower, upper))
    """
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_labels = np.asarray(y_labels, dtype=np.int32)
    valid_mask = y_labels != -1
    y_probs = y_probs[valid_mask]
    y_labels = y_labels[valid_mask]

    if y_labels.size == 0:
        zero_ci = (0.0, (0.0, 0.0))
        return {k: zero_ci for k in [
            "accuracy", "precision", "recall", "f1",
            "auroc", "auprc", "sensitivity", "specificity", "youden"
        ]}

    rng = np.random.default_rng(seed)
    n = y_labels.size
    metrics_samples = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "auroc": [], "auprc": [], "sensitivity": [], "specificity": [], "youden": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        probs_s = y_probs[idx]
        labels_s = y_labels[idx]
        preds_s = (probs_s >= float(threshold)).astype(int)

        tp = np.sum((preds_s == 1) & (labels_s == 1))
        tn = np.sum((preds_s == 0) & (labels_s == 0))
        fp = np.sum((preds_s == 1) & (labels_s == 0))
        fn = np.sum((preds_s == 0) & (labels_s == 1))

        acc = float((preds_s == labels_s).mean())
        prec = precision_score(labels_s, preds_s, zero_division=0)
        rec = recall_score(labels_s, preds_s, zero_division=0)
        f1 = f1_score(labels_s, preds_s, zero_division=0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0

        try:
            auroc = roc_auc_score(labels_s, probs_s) if len(np.unique(labels_s)) > 1 else float("nan")
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(labels_s, probs_s) if len(np.unique(labels_s)) > 1 else float("nan")
        except Exception:
            auprc = float("nan")

        metrics_samples["accuracy"].append(acc)
        metrics_samples["precision"].append(float(prec))
        metrics_samples["recall"].append(float(rec))
        metrics_samples["f1"].append(float(f1))
        metrics_samples["sensitivity"].append(float(sens))
        metrics_samples["specificity"].append(float(spec))
        metrics_samples["youden"].append(float(youden))
        metrics_samples["auroc"].append(float(np.nan if np.isnan(auroc) else auroc))
        metrics_samples["auprc"].append(float(np.nan if np.isnan(auprc) else auprc))

    results = {}
    alpha = 1.0 - ci
    for k, vals in metrics_samples.items():
        arr = np.asarray(vals, dtype=np.float32)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            results[k] = (0.0, (0.0, 0.0))
            continue
        mean = float(arr_valid.mean())
        lower = float(np.percentile(arr_valid, 100 * alpha / 2))
        upper = float(np.percentile(arr_valid, 100 * (1 - alpha / 2)))
        results[k] = (mean, (lower, upper))

    return results


def load_agent_results(results_path: str) -> tuple[dict[str, str], dict[str, float]]:
    """
    加载 Agent 推理结果 JSON。
    返回: (image_name -> predicted_class, image_name -> P(恶性))
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = {}
    probs = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        image_name = item.get("image_name") or item.get("filename")
        if not image_name:
            continue

        pred_class = item.get("predicted_class", "")
        confidence = float(item.get("confidence", 0.5))

        predictions[image_name] = pred_class

        # P(恶性): 若预测恶性则 confidence 即为恶性概率；若预测良性则 1 - confidence
        if pred_class == "恶性":
            probs[image_name] = confidence
        elif pred_class == "良性":
            probs[image_name] = 1.0 - confidence
        else:
            probs[image_name] = 0.5

    return predictions, probs


def load_labels(label_path: str) -> dict[str, int]:
    """
    加载标签文件。支持 JSON 数组格式 {filename, malignancy}。
    malignancy: 0=良性, 1=恶性
    返回: image_name -> 0|1
    """
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or "filename" not in item or "malignancy" not in item:
                continue
            fn = item["filename"]
            labels[fn] = int(item["malignancy"])
    elif isinstance(data, dict):
        for k, v in data.items():
            labels[k] = int(v)

    return labels


def compute_metrics(
    results_path: str,
    label_path: str,
    threshold: float = 0.5,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
):
    """
    计算分类指标及 CI95。
    """
    _, probs = load_agent_results(results_path)
    labels = load_labels(label_path)

    # 按 image_name 对齐
    common = sorted(set(probs.keys()) & set(labels.keys()))
    if not common:
        raise ValueError(
            f"结果与标签无交集。结果样本数: {len(probs)}, 标签样本数: {len(labels)}"
        )

    y_prob = np.array([probs[n] for n in common], dtype=np.float32)
    y_true = np.array([labels[n] for n in common], dtype=np.int32)

    metrics_ci = classification_bootstrap_metrics(
        y_probs=y_prob,
        y_labels=y_true,
        threshold=threshold,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )

    # 映射: sensitivity = recall
    return {
        "AUROC": metrics_ci["auroc"],
        "AUPRC": metrics_ci["auprc"],
        "Acc": metrics_ci["accuracy"],
        "Prec": metrics_ci["precision"],
        "Recall": metrics_ci["recall"],
        "F1": metrics_ci["f1"],
        "Specificity": metrics_ci["specificity"],
    }, len(common)


def main():
    cfg = CONFIG
    results_path = Path(cfg["results_path"])
    label_path = Path(cfg["label_path"])

    if not results_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    metrics, n = compute_metrics(
        str(results_path),
        str(label_path),
        threshold=cfg["threshold"],
        n_boot=cfg["n_boot"],
        ci=cfg["ci"],
        seed=cfg["seed"],
    )

    # 打印
    print(f"\n样本数: {n}")
    print("-" * 50)
    for name, (mean, (lo, hi)) in metrics.items():
        print(f"{name:12} {mean:.4f}  [CI95: {lo:.4f}, {hi:.4f}]")
    print("-" * 50)

    # 输出 JSON
    out = {
        "n_samples": n,
        "threshold": cfg["threshold"],
        "n_boot": cfg["n_boot"],
        "ci": cfg["ci"],
        "metrics": {
            k: {"mean": round(v[0], 4), "ci95_lower": round(v[1][0], 4), "ci95_upper": round(v[1][1], 4)}
            for k, v in metrics.items()
        },
    }

    if cfg.get("output_path"):
        out_path = Path(cfg["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {out_path}")

    return out


if __name__ == "__main__":
    main()
