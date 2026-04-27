"""
根据 Agent 最终输出的 results_*.json 与标签文件，计算分类 Agent 的平均性能。

支持：
- 结果 JSON：main.py 输出的格式（image_name, predicted_class, confidence）
- 标签 JSON：列表 [{"filename": "xx.jpg", "malignancy": 0|1}] 或 字典 {"xx.jpg": 0|1}，malignancy 0=良性 1=恶性
- 预测类别名：中文 良性/恶性 或 英文 benign/malignant 均可

指标：AUROC, AUPRC, Acc, Prec, Recall, F1, Specificity, ECE（含 bootstrap CI95）

用法：修改下方 CONFIG 后直接运行 python scripts/compute_agent_metrics_from_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============ 配置（直接修改此处） ============
CONFIG = {
    "results_path": "output/TN3K_test/results_20260114_121753.json",  # Agent 输出 JSON
    "label_path": "test_label.json",                                 # 标签 JSON
    "output_path": "output/TN3K_test/agent_metrics.json",            # 指标输出 JSON（None 则只打印）
    "threshold": 0.5,
    "ci": True,       # 默认计算 bootstrap CI95
    "n_boot": 2000,
    "seed": 0,
}
# =============================================


# 预测类别到“恶性”的映射（用于统一成 P(恶性)）
MALIGNANT_ALIASES = {"恶性", "malignant", "1"}
BENIGN_ALIASES = {"良性", "benign", "0"}


def load_results(results_path: str) -> tuple[dict[str, str], dict[str, float], dict[str, int], int]:
    """
    加载 Agent 输出 JSON（demo 生成的 results_*.json）。
    返回: (
        image_name -> predicted_class,
        image_name -> P(恶性),
        selected_model -> count,
        含 selected_model 字段的样本数
    )
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("results", data.get("decisions", []))
    if not isinstance(data, list):
        data = [data]

    predictions = {}
    probs = {}
    selected_model_counts = {}
    selected_model_samples = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        image_name = item.get("image_name") or item.get("filename")
        if not image_name:
            continue
        pred_class = (item.get("predicted_class") or item.get("selected_class") or "").strip()
        confidence = float(item.get("confidence", 0.5))

        predictions[image_name] = pred_class
        pred_lower = pred_class.lower()
        if pred_lower in {a.lower() for a in MALIGNANT_ALIASES} or pred_class == "恶性":
            probs[image_name] = confidence
        elif pred_lower in {a.lower() for a in BENIGN_ALIASES} or pred_class == "良性":
            probs[image_name] = 1.0 - confidence
        else:
            probs[image_name] = 0.5

        selected_model = item.get("selected_model")
        if selected_model is not None and str(selected_model).strip():
            selected_model = str(selected_model).strip()
            selected_model_counts[selected_model] = selected_model_counts.get(selected_model, 0) + 1
            selected_model_samples += 1

    return predictions, probs, selected_model_counts, selected_model_samples


def load_labels(label_path: str) -> dict[str, int]:
    """
    加载标签 JSON。
    支持: [{"filename": "xx.jpg", "malignancy": 0|1}] 或 {"xx.jpg": 0|1}
    malignancy: 0=良性, 1=恶性
    返回: image_name -> 0|1
    """
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fn = item.get("filename") or item.get("image_name")
            if fn is None:
                continue
            mal = item.get("malignancy", item.get("label"))
            if mal is not None:
                labels[fn] = int(mal)
    elif isinstance(data, dict):
        for k, v in data.items():
            labels[k] = int(v)
    return labels


def compute_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """计算点估计指标（无 bootstrap）。"""
    y_pred_bin = (y_prob >= threshold).astype(np.int32)
    n = len(y_true)
    if n == 0:
        return {
            "acc": 0.0,
            "prec": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "specificity": 0.0,
            "ece": float("nan"),
        }

    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    tn = np.sum((y_pred_bin == 0) & (y_true == 0))
    fp = np.sum((y_pred_bin == 1) & (y_true == 0))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    ece = compute_ece_binary(y_true, y_prob, n_bins=10)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "specificity": float(spec),
        "ece": float(ece) if not np.isnan(ece) else float("nan"),
    }


def compute_ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: sum_{bins} (bin_prob * |acc_bin - conf_bin|)."""
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf_bin = float(np.mean(y_prob[mask]))
        acc_bin = float(np.mean(y_true[mask]))
        bin_prob = float(np.mean(mask))
        ece += bin_prob * abs(acc_bin - conf_bin)
    return float(ece)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """Bootstrap 得到各指标均值和 CI。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    names = ["auroc", "auprc", "acc", "prec", "recall", "f1", "specificity", "ece"]
    samples = {k: [] for k in names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        m = compute_point_metrics(yt, yp, threshold=threshold)
        for k in names:
            samples[k].append(float(m[k]) if not np.isnan(m[k]) else np.nan)

    alpha = 1.0 - ci
    out = {}
    for k in names:
        arr = np.array(samples[k], dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            out[k] = (float("nan"), (float("nan"), float("nan")))
        else:
            mean = float(np.mean(valid))
            lo = float(np.percentile(valid, 100 * alpha / 2))
            hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
            out[k] = (mean, (lo, hi))
    return out


def main():
    cfg = CONFIG
    results_path = Path(cfg["results_path"])
    label_path = Path(cfg["label_path"])

    if not results_path.exists():
        print(f"错误: 结果文件不存在: {results_path}", file=sys.stderr)
        sys.exit(1)
    if not label_path.exists():
        print(f"错误: 标签文件不存在: {label_path}", file=sys.stderr)
        sys.exit(1)

    _, probs, selected_model_counts, selected_model_samples = load_results(str(results_path))
    labels = load_labels(str(label_path))
    common = sorted(set(probs.keys()) & set(labels.keys()))
    if not common:
        print(
            f"错误: 结果与标签无交集。结果样本数: {len(probs)}, 标签样本数: {len(labels)}",
            file=sys.stderr,
        )
        sys.exit(1)

    y_true = np.array([labels[n] for n in common], dtype=np.int32)
    y_prob = np.array([probs[n] for n in common], dtype=np.float64)
    threshold = cfg["threshold"]

    # 点估计
    point = compute_point_metrics(y_true, y_prob, threshold)

    print("=" * 60)
    print("分类 Agent 平均性能（与标签对齐样本数: {}）".format(len(common)))
    print("=" * 60)
    print(f"{'指标':<14} {'均值':>10}")
    print("-" * 26)
    for name, key in [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Acc", "acc"),
        ("Prec", "prec"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Specificity", "specificity"),
        ("ECE", "ece"),
    ]:
        v = point[key]
        if np.isnan(v):
            print(f"{name:<14} {'N/A':>10}")
        else:
            print(f"{name:<14} {v:>10.4f}")
    print("-" * 26)

    report = {
        "n_samples": len(common),
        "threshold": threshold,
        "metrics": {k: round(v, 4) if not (isinstance(v, float) and np.isnan(v)) else None for k, v in point.items()},
        "selected_model_stats": {
            "n_samples_with_selected_model": selected_model_samples,
            "model_counts": selected_model_counts,
            "model_ratios": {
                k: round(v / selected_model_samples, 4) for k, v in selected_model_counts.items()
            } if selected_model_samples > 0 else {},
        },
    }

    if cfg.get("ci"):
        ci_results = compute_bootstrap_ci(
            y_true, y_prob, threshold=threshold, n_boot=cfg.get("n_boot", 2000), ci=0.95, seed=cfg.get("seed", 0)
        )
        print("\nBootstrap CI95:")
        for name, key in [
            ("AUROC", "auroc"),
            ("AUPRC", "auprc"),
            ("Acc", "acc"),
            ("Prec", "prec"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("Specificity", "specificity"),
            ("ECE", "ece"),
        ]:
            mean, (lo, hi) = ci_results[key]
            if np.isnan(mean):
                print(f"  {name}: N/A")
            else:
                print(f"  {name}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
        report["metrics_ci95"] = {
            k: {
                "mean": None if np.isnan(v[0]) else round(v[0], 4),
                "ci95_lower": None if np.isnan(v[1][0]) else round(v[1][0], 4),
                "ci95_upper": None if np.isnan(v[1][1]) else round(v[1][1], 4),
            }
            for k, v in ci_results.items()
        }
        report["n_boot"] = cfg.get("n_boot", 2000)

    output_path = cfg.get("output_path")
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
