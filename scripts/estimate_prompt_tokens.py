"""
估算单次 Agent 调用的 prompt 输入量（字符数 + 约 token 数）
用法: python scripts/estimate_prompt_tokens.py [--config config/config.yaml]

Prompt 组成（与 agent/classification_agent.py select_best_model 一致）：
  - system_prompt（固定）
  - device_info_text（来自 config data.device_info）
  - base_datasets_text（来自 config base_datasets_info）
  - 固定引导句 + format_predictions_json(predictions) + 结尾要求与 JSON 示例
"""
import json
import math
import sys
import yaml
import argparse
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.base_model import ModelOutput


# 与 classification_agent.LLMClassificationAgent.system_prompt 一致（短版），仅用于统计长度
SYSTEM_PROMPT = """你是甲状腺超声多模型预测整合专家，从若干模型输出中选最可信的一项。

【设备】设备决定成像风格；训练数据覆盖输入同款/同品牌者更可信。GE(Logiq E9/S7 等)与 Hitachi(ARIETTA 等)各系内部风格近；其余品牌与上述有差异。Heterogeneous=多设备混合。输入设备未知则忽略此项。

【字段】主置信度优先 metadata.classification_uncertainty.top_confidence_calibrated，否则 top_confidence_raw 或 top_confidence。entropy(越大越不确定)、margin_top2(越大越稳) 在同路径下。consistency_metrics：num_models_same_class、total_models、vote_entropy。

【决策序】1)主置信度 2)已知输入设备则设备匹配 3)主置信度差<0.05 时比 validation_metrics.on_training_dataset 的 acc/AUC/F1 4)仍接近则 entropy 更低、margin_top2 更高 5)能推断 TN3K/ThyroidXL/TN5K/CineClip 则看 base_dataset_performance，否则 dataset_size 更大优先 6)差<0.05 结合投票与 num_models_same_class；类别冲突时若有单模型主置信度>0.95 可优先 7)仅当差<0.02 再考虑模型结构差异。

【输出】只输出纯 JSON（无 Markdown/思考/代码块），首尾为 { }。键：selected_model, selected_class, confidence(0~1), reasoning。reasoning 用中文、客观简练，须含主置信度数值、entropy/margin_top2、与 1~2 个次选的关键数值对比（可含设备/验证指标/一致性），忌空话修辞。"""


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_predictions_json_like_agent(predictions: list) -> str:
    """与 LLMClassificationAgent.format_predictions_json（压缩版）逻辑一致，用于估算长度"""
    votes_per_class = {}
    for pred in predictions:
        votes_per_class[pred.top_class] = votes_per_class.get(pred.top_class, 0) + 1
    total_models = len(predictions)
    vote_entropy = 0.0
    if total_models > 0:
        probs = [c / total_models for c in votes_per_class.values()]
        vote_entropy = -sum(p * math.log(p + 1e-12, 2) for p in probs if p > 0)

    predictions_data = []
    for pred in predictions:
        pred_dict = pred.to_dict()
        metadata = pred_dict.get("metadata") or {}
        full_probs = pred_dict.get("predictions", {}) or {}
        class_probs = list(full_probs.values())
        entropy = None
        margin_top2 = None
        if class_probs:
            total_prob = sum(class_probs)
            if total_prob > 0:
                norm_probs = [p / total_prob for p in class_probs]
                entropy = -sum(p * math.log(p + 1e-12, 2) for p in norm_probs if p > 0)
            sorted_probs = sorted(class_probs, reverse=True)
            margin_top2 = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else sorted_probs[0]
        sorted_items = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_predictions = [{k: float(v)} for k, v in sorted_items[:2]]

        cu = metadata.get("classification_uncertainty", {}) or {}
        on_train = (metadata.get("validation_metrics", {}) or {}).get("on_training_dataset", {}) or {}
        dataset_info = metadata.get("dataset_info", {}) or {}
        base_perf = metadata.get("base_dataset_performance", {}) or {}

        predictions_data.append({
            "model_name": pred_dict.get("model_name"),
            "top_class": pred_dict.get("top_class"),
            "top_confidence": pred_dict.get("top_confidence"),
            "top2_predictions": top_k_predictions,
            "metadata": {
                "classification_uncertainty": {
                    **(
                        {"top_confidence_calibrated": cu.get("top_confidence_calibrated")}
                        if cu.get("top_confidence_calibrated") is not None
                        else {}
                    ),
                    "top_confidence_raw": pred_dict.get("top_confidence"),
                    "entropy": entropy,
                    "margin_top2": margin_top2,
                },
                "consistency_metrics": {
                    "num_models_same_class": votes_per_class.get(pred_dict.get("top_class"), 0),
                    "total_models": total_models,
                    "vote_entropy": vote_entropy,
                },
                "training_data_devices": metadata.get("training_data_devices") or [],
                "dataset_info": {
                    "training_dataset": dataset_info.get("training_dataset"),
                    "base_datasets": dataset_info.get("base_datasets", []),
                    "dataset_size": dataset_info.get("dataset_size"),
                },
                "validation_metrics": {
                    "on_training_dataset": {
                        "accuracy": on_train.get("accuracy"),
                        "auc": on_train.get("auc"),
                        "f1_score": on_train.get("f1_score"),
                    }
                },
                "base_dataset_performance": base_perf,
            },
        })
    return json.dumps({"num_models": len(predictions), "predictions": predictions_data}, ensure_ascii=False)  # 与 agent 一致，不缩进


def build_mock_predictions_from_config(config: dict) -> list:
    """根据 config 中 dino_unet 与 autogluon 的模型配置，构造与真实推理等价的 ModelOutput 列表（仅用于估算 JSON 长度）"""
    predictions = []
    # 二分类典型键名
    class_a, class_b = "benign", "malignant"

    # DINO-UNet: 5 个模型
    for m in config.get("dino_unet", {}).get("models", []):
        name = m.get("name", "dino")
        base_perf = m.get("base_dataset_performance", {})
        ds_info = m.get("dataset_info", {})
        base_sets = ds_info.get("base_datasets", [])
        size = ds_info.get("dataset_size", 0)
        # 设备从 base_datasets_info 推导，这里用占位
        devices = []
        base_info = config.get("base_datasets_info", {})
        for b in base_sets:
            if b in base_info and isinstance(base_info[b], dict):
                devices.extend(base_info[b].get("main_devices", []))
        devices = list(dict.fromkeys(devices))  # 去重保序

        metadata = {
            "framework": "pytorch",
            "training_data_devices": devices,
            "dataset_info": {
                "training_dataset": name,
                "base_datasets": base_sets,
                "dataset_size": size,
            },
            "validation_metrics": {
                "on_training_dataset": {"accuracy": 0.85, "auc": 0.88, "f1_score": 0.84},
            },
            "base_dataset_performance": base_perf,
        }
        pred = ModelOutput(
            model_name=name,
            predictions={class_a: 0.65, class_b: 0.35},
            top_class=class_a,
            top_confidence=0.65,
            requires_mask=True,
            metadata=metadata,
        )
        predictions.append(pred)

    # AutoGluon: 1 个模型
    for m in config.get("autogluon", {}).get("models", []):
        name = m.get("name", "autogluon")
        base_perf = m.get("base_dataset_performance", {})
        ds_info = m.get("dataset_info", {})
        base_sets = ds_info.get("base_datasets", [])
        size = ds_info.get("dataset_size", 0)
        devices = []
        base_info = config.get("base_datasets_info", {})
        for b in base_sets:
            if b in base_info and isinstance(base_info[b], dict):
                devices.extend(base_info[b].get("main_devices", []))
        devices = list(dict.fromkeys(devices))

        metadata = {
            "framework": "AutoGluon + PyRadiomics",
            "training_data_devices": devices,
            "dataset_info": {
                "training_dataset": name,
                "base_datasets": base_sets,
                "dataset_size": size,
            },
            "validation_metrics": {
                "on_training_dataset": {"accuracy": 0.87, "auc": 0.90, "f1_score": 0.86},
            },
            "base_dataset_performance": base_perf,
        }
        pred = ModelOutput(
            model_name=name,
            predictions={class_a: 0.72, class_b: 0.28},
            top_class=class_a,
            top_confidence=0.72,
            requires_mask=False,
            metadata=metadata,
        )
        predictions.append(pred)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="估算 Agent 单次调用的 prompt 输入量")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "config.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # 构造与真实调用等价的 prompt（不发请求）
    mock_predictions = build_mock_predictions_from_config(config)
    formatted_preds = format_predictions_json_like_agent(mock_predictions)

    device_info = config.get("data", {}).get("device_info")
    if device_info:
        device_info_text = "\n输入设备: " + ", ".join(device_info) + "\n"
    else:
        device_info_text = "\n输入设备: 未知\n"

    base_datasets_text = ""
    base_info = config.get("base_datasets_info") or {}
    if base_info:
        base_datasets_text = "\n数据集→设备(推断来源):\n"
        for ds_name, ds_dict in base_info.items():
            if isinstance(ds_dict, dict) and ds_dict.get("main_devices"):
                base_datasets_text += f"- {ds_name}: {', '.join(ds_dict['main_devices'])}\n"

    n = len(mock_predictions)
    tail = f"""
以下为 {n} 个模型的预测(JSON)：

{formatted_preds}

选出最佳结果，严格按【输出】只回复 JSON。"""

    full_prompt = SYSTEM_PROMPT + device_info_text + base_datasets_text + tail

    char_count = len(full_prompt)
    # 中英混合常见估算：约 1.5~2 字符/token，这里取 1.6
    estimated_tokens = int(char_count / 1.6)

    print("=" * 60)
    print("Prompt 输入量估算（单张图像、单次 Agent 调用）")
    print("=" * 60)
    print(f"  配置: {config_path.name}")
    print(f"  模型数: {n} (dino_unet + autogluon)")
    print()
    print("  组成:")
    print(f"    - system_prompt:           {len(SYSTEM_PROMPT):>6} 字符")
    print(f"    - device_info:             {len(device_info_text):>6} 字符")
    print(f"    - base_datasets 映射:      {len(base_datasets_text):>6} 字符")
    print(f"    - 预测 JSON + 结尾说明:    {len(tail):>6} 字符")
    print()
    print(f"  总字符数:   {char_count}")
    print(f"  约 token 数: ~{estimated_tokens} （按 1.6 字符/token 估算，中英混合）")
    print()
    print("  说明: 实际 API 计费以服务端 token 统计为准；批量模式每张图会再乘图像数。")
    print("=" * 60)


if __name__ == "__main__":
    main()
