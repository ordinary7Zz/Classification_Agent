"""
甲状腺分类模型主入口
展示 DINO-UNet 和 AutoGluon-PyRadiomics 两个模型的使用
支持单个文件或批量处理目录
"""

import os
import sys
import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_registry import ModelRegistry
from models.dino_unet_model import DINOUNetModel
from models.autogluon_radiomics_model import AutoGluonRadiomicsModel
from agent.classification_agent import (
    LLMClassificationAgent,
    _average_class_probabilities,
    _winning_class_from_avg_probs,
)
from utils.image_processor import ImageProcessor
from calibration.runtime import (
    load_calibration_map_from_config,
    maybe_apply_calibration_map,
)


def load_config(config_path: str = "config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_files(directory: str) -> List[Path]:
    """获取目录中所有图像文件"""
    directory = Path(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))

    unique_files = sorted(set(f.resolve() for f in image_files))
    return unique_files


def find_corresponding_mask(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """在掩码目录中查找与图像文件名对应的掩码文件"""
    mask_path = mask_dir / image_path.name
    if mask_path.exists():
        return mask_path

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for ext in image_extensions:
        mask_path = mask_dir / f"{image_path.stem}{ext}"
        if mask_path.exists():
            return mask_path

    return None


def derive_training_data_devices(
    dataset_info: Optional[dict],
    base_datasets_info: Optional[dict]
) -> List[str]:
    """
    从base_datasets和base_datasets_info推导training_data_devices
    """
    if not dataset_info or 'base_datasets' not in dataset_info:
        return []

    if not base_datasets_info:
        return []

    base_datasets = dataset_info['base_datasets']
    if not base_datasets:
        return []

    all_devices = set()
    for base_dataset in base_datasets:
        if base_dataset in base_datasets_info:
            dataset_info_dict = base_datasets_info[base_dataset]
            if isinstance(dataset_info_dict, dict) and 'main_devices' in dataset_info_dict:
                devices = dataset_info_dict['main_devices']
                if devices:
                    all_devices.update(devices)

    return sorted(list(all_devices))


def infer_label_path_by_output_dir(output_dir: str) -> Optional[Path]:
    """按 output_dir 名称推断标签文件路径。"""
    out_name = Path(output_dir).name
    project_root = Path(__file__).resolve().parent

    candidates: list[Path] = []
    if "TN3K" in out_name:
        candidates = [
            project_root / "tn3k_test_label.json",
            project_root / "TN3K_test_label.json",
            project_root / "dataset" / "tn3k_test_label.json",
            project_root / "dataset" / "TN3K_test_label.json",
        ]
    elif "TN5K" in out_name:
        candidates = [
            project_root / "TN5K_test_label.json",
            project_root / "dataset" / "TN5K_test_label.json",
        ]
    elif "ThyroidXL" in out_name:
        candidates = [
            project_root / "ThyroidXL_test_label.json",
            project_root / "dataset" / "ThyroidXL_test_label.json",
        ]
    else:
        candidates = [
            project_root / "test_label.json",
            project_root / "dataset" / "test_label.json",
        ]

    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_label_path(paths_config: dict) -> tuple[Optional[Path], str]:
    """
    解析标签文件路径。
    返回: (label_path_or_none, source)
      - source: "config" | "auto" | "none"
    """
    data_cfg = paths_config.get("data", {}) if isinstance(paths_config, dict) else {}
    configured_label = data_cfg.get("label_file", None)
    project_root = Path(__file__).resolve().parent

    if configured_label is not None:
        configured_label_str = str(configured_label).strip()
        if configured_label_str and configured_label_str.lower() != "null":
            configured_path = Path(configured_label_str)
            if not configured_path.is_absolute():
                configured_path = project_root / configured_path
            if configured_path.exists():
                return configured_path, "config"
            return None, "config"

    output_dir = str(paths_config.get("output", {}).get("output_dir", "output"))
    auto_path = infer_label_path_by_output_dir(output_dir)
    if auto_path is not None and auto_path.exists():
        return auto_path, "auto"
    return None, "none"


def _load_labels(label_path: Path) -> dict[str, int]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: dict[str, int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fn = item.get("filename") or item.get("image_name")
            mal = item.get("malignancy", item.get("label"))
            if fn is None or mal is None:
                continue
            labels[str(fn)] = int(mal)
    elif isinstance(data, dict):
        for k, v in data.items():
            labels[str(k)] = int(v)
    return labels


def _label_lookup(labels: dict[str, int], image_name: str):
    if image_name in labels:
        return labels[image_name]

    ext = Path(image_name).suffix
    stem = Path(image_name).stem
    tokens = stem.split("_")
    if len(tokens) >= 2:
        candidate = tokens[-1] + ext
        if candidate in labels:
            return labels[candidate]
    return None


def _infer_malignant_probability(predicted_class: str, confidence: float) -> float:
    pred_class = str(predicted_class).strip()
    pred_lower = pred_class.lower()
    if pred_class in {"恶性", "malignant", "1"} or pred_lower in {"恶性", "malignant", "1"}:
        return float(confidence)
    if pred_class in {"良性", "benign", "0"} or pred_lower in {"良性", "benign", "0"}:
        return float(1.0 - confidence)
    return 0.5


def _build_result_dict(
    image_file: Path,
    selected_model: str,
    predicted_class: str,
    confidence: float,
    reasoning: str,
    predictions: list,
    labels: Optional[dict[str, int]],
    decision_threshold: float,
) -> dict:
    image_name = image_file.name
    ground_truth_label = _label_lookup(labels, image_name) if labels is not None else None
    malignant_probability = _infer_malignant_probability(predicted_class, confidence)

    return {
        "image_file": str(image_file),
        "image_name": image_name,
        "selected_model": selected_model,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "malignant_probability": float(malignant_probability),
        "ground_truth_label": None if ground_truth_label is None else int(ground_truth_label),
        "decision_threshold": float(decision_threshold),
        "reasoning": reasoning,
        "all_predictions": [
            {
                "model": p.model_name,
                "top_class": p.top_class,
                "top_confidence": float(p.top_confidence),
                "predictions": {k: float(v) for k, v in p.predictions.items()}
            }
            for p in predictions
        ]
    }


def main(config_path: str = "config/config.yaml"):
    print("=" * 70)
    print("甲状腺结节分类 Agent")
    print("=" * 70)
    print()

    print(">>> 加载配置")
    config = load_config(config_path)
    if config is None:
        print("✗ 请先配置 config/config.yaml 文件")
        return

    paths_config = config
    print("✓ 配置加载成功\n")

    print(">>> 检查标签文件")
    resolved_label_path, label_path_source = resolve_label_path(paths_config)
    if resolved_label_path is not None:
        if label_path_source == "config":
            print(f"✓ 已扫描到标签文件（来自配置）: {resolved_label_path}\n")
        else:
            print(f"✓ 已扫描到标签文件（自动推断）: {resolved_label_path}\n")
    else:
        configured_label = paths_config.get("data", {}).get("label_file", None)
        if configured_label is None or str(configured_label).strip().lower() == "null":
            print("⚠️ 未扫描到标签文件（config.data.label_file=null，且自动推断未命中）\n")
        else:
            print(f"⚠️ 未扫描到标签文件（配置路径不存在）: {configured_label}\n")

    registry = ModelRegistry()

    print(">>> 注册 DINO-UNet 多任务模型")
    dino_models_config = paths_config['dino_unet'].get('models', [])

    if not dino_models_config:
        print("⚠️  没有配置 DINO-UNet 模型\n")
    else:
        print(f"    找到 {len(dino_models_config)} 个 DINO-UNet 模型配置")

        for idx, model_config in enumerate(dino_models_config, 1):
            model_name = model_config.get('name', f'dino_unet_{idx}')
            model_path = model_config['model_path']
            use_tirads = model_config.get('use_tirads', False)

            print(f"\n    [{idx}] {model_name}")
            print(f"        路径: {model_path}")
            print(f"        TI-RADS: {use_tirads}")

            if not os.path.exists(model_path):
                print(f"        ⚠️  模型路径不存在，跳过")
                continue

            try:
                dino_model = DINOUNetModel(
                    model_path=model_path,
                    device="cuda",
                    use_tirads=use_tirads
                )
                dino_model.load_model()

                if 'name' in model_config:
                    dino_model.model_name = f"DINO_UNet_{model_name}"

                dataset_info = model_config.get('dataset_info', None)
                if dataset_info:
                    dino_model.dataset_info = dataset_info

                training_data_devices = model_config.get('training_data_devices', None)
                if training_data_devices is None or len(training_data_devices) == 0:
                    base_datasets_info = paths_config.get('base_datasets_info', None)
                    training_data_devices = derive_training_data_devices(
                        dataset_info, base_datasets_info
                    )
                    if training_data_devices:
                        print(f"        自动推导训练设备: {', '.join(training_data_devices)}")
                dino_model.training_data_devices = training_data_devices

                validation_metrics = model_config.get('validation_metrics', None)
                if validation_metrics:
                    dino_model.validation_metrics = validation_metrics

                base_dataset_performance = model_config.get('base_dataset_performance', None)
                if base_dataset_performance:
                    dino_model.base_dataset_performance = base_dataset_performance

                registry.register_model(dino_model)
                print(f"        ✓ 注册成功")
            except Exception as e:
                print(f"        ✗ 注册失败: {e}")

        print()

    print(">>> 注册 AutoGluon-PyRadiomics 模型")
    autogluon_models_config = paths_config['autogluon'].get('models', [])

    if not autogluon_models_config:
        print("⚠️  没有配置 AutoGluon 模型\n")
    else:
        print(f"    找到 {len(autogluon_models_config)} 个 AutoGluon 模型配置")

        for idx, model_config in enumerate(autogluon_models_config, 1):
            model_name = model_config.get('name', f'autogluon_{idx}')
            model_dir = model_config['model_dir']

            print(f"\n    [{idx}] {model_name}")
            print(f"        目录: {model_dir}")

            if not os.path.exists(model_dir):
                print(f"        ⚠️  模型目录不存在，跳过")
                continue

            try:
                autogluon_model = AutoGluonRadiomicsModel(
                    model_dir=model_dir
                )
                autogluon_model.load_model()

                if 'name' in model_config:
                    autogluon_model.model_name = f"AutoGluon_{model_name}"

                dataset_info = model_config.get('dataset_info', None)
                if dataset_info:
                    autogluon_model.dataset_info = dataset_info

                training_data_devices = model_config.get('training_data_devices', None)
                if training_data_devices is None or len(training_data_devices) == 0:
                    base_datasets_info = paths_config.get('base_datasets_info', None)
                    training_data_devices = derive_training_data_devices(
                        dataset_info, base_datasets_info
                    )
                    if training_data_devices:
                        print(f"        自动推导训练设备: {', '.join(training_data_devices)}")
                autogluon_model.training_data_devices = training_data_devices

                validation_metrics = model_config.get('validation_metrics', None)
                if validation_metrics:
                    autogluon_model.validation_metrics = validation_metrics

                base_dataset_performance = model_config.get('base_dataset_performance', None)
                if base_dataset_performance:
                    autogluon_model.base_dataset_performance = base_dataset_performance

                registry.register_model(autogluon_model)
                print(f"        ✓ 注册成功")
            except Exception as e:
                print(f"        ✗ 注册失败: {e}")

        print()

    if len(registry.list_models()) == 0:
        print("没有可用的模型，退出。")
        return

    project_root = Path(__file__).resolve().parent
    try:
        cal_map = load_calibration_map_from_config(config, project_root)
        registry.calibration_map = cal_map if cal_map else None
        if cal_map:
            print(
                f">>> 已加载概率校准表: {len(cal_map)} 个模型 {list(cal_map.keys())}\n"
            )
    except Exception:
        registry.calibration_map = None

    print(">>> 初始化 Agent LLM")
    base_datasets_info = paths_config.get('base_datasets_info', None)
    agent_config = config.get('agent', {})
    max_batch_size = agent_config.get('max_batch_size', 10)
    enable_agent = agent_config.get('enable_agent', True)
    top_k = max(1, int(agent_config.get('top_k', 1)))

    agent = None
    if enable_agent:
        agent = LLMClassificationAgent(
            api_key=config['agent_llm']['api_key'],
            model_name=config['agent_llm']['model_name'],
            temperature=config['agent_llm']['temperature'],
            base_datasets_info=base_datasets_info,
            max_batch_size=max_batch_size,
            selection_mode=agent_config.get('selection_mode', 'deterministic'),
            top_k=top_k,
        )
        print(f"✓ Agent LLM 初始化成功 (max_batch_size={max_batch_size}, top_k={top_k})\n")
    else:
        print(
            f"⚠️ 已在配置中关闭 Agent LLM，将按 top_confidence 取前 {top_k} 个模型做 soft voting。\n"
        )

    image_processor = ImageProcessor()

    print(">>> 加载图像路径")
    image_input = paths_config['data']['image_input']

    print(f"    图像路径: {image_input}")

    if not os.path.exists(image_input):
        print(f"✗ 路径不存在: {image_input}")
        print("   请检查 config/config.yaml 中的 data.image_input 配置")
        return

    image_input_path = Path(image_input)
    data_config = paths_config.get('data', {}) if isinstance(paths_config, dict) else {}
    raw_start_index = data_config.get('start_index', 0)
    raw_total_count = data_config.get('total_count', None)

    try:
        start_index = int(raw_start_index or 0)
    except (TypeError, ValueError):
        print(f"✗ data.start_index 配置无效: {raw_start_index}")
        print("   start_index 必须是大于等于 0 的整数")
        return

    if start_index < 0:
        print(f"✗ data.start_index 配置无效: {start_index}")
        print("   start_index 必须是大于等于 0 的整数")
        return

    total_count = None
    if raw_total_count is not None and str(raw_total_count).strip().lower() != 'null':
        try:
            total_count = int(raw_total_count)
        except (TypeError, ValueError):
            print(f"✗ data.total_count 配置无效: {raw_total_count}")
            print("   total_count 必须是正整数或 null")
            return
        if total_count <= 0:
            print(f"✗ data.total_count 配置无效: {total_count}")
            print("   total_count 必须是正整数或 null")
            return

    if image_input_path.is_file():
        if start_index not in {0} or (total_count is not None and total_count != 1):
            print("✗ 当前 data.image_input 是单个文件，start_index/total_count 切片配置仅对目录输入有意义")
            print("   单文件输入时仅允许 start_index=0，且 total_count 为 null 或 1")
            return
        image_files = [image_input_path]
        print(f"✓ 检测到单个图像文件")
    elif image_input_path.is_dir():
        image_files = get_image_files(image_input)
        total_images = len(image_files)
        print(f"✓ 检测到图像目录，找到 {total_images} 个图像文件")

        if total_images == 0:
            print("✗ 未找到任何图像文件")
            return

        if start_index >= total_images:
            print(f"✗ data.start_index 超出范围: {start_index}")
            print(f"   当前目录共有 {total_images} 个图像文件，有效索引范围为 0 到 {total_images - 1}")
            return

        if total_count is None:
            end_index = total_images
            image_files = image_files[start_index:]
            total_count_display = '直到末尾'
        else:
            end_index = min(start_index + total_count, total_images)
            image_files = image_files[start_index:end_index]
            total_count_display = str(total_count)

        print("    数据切片配置:")
        print(f"      原始总数: {total_images}")
        print(f"      start_index: {start_index}")
        print(f"      total_count: {total_count_display}")
        print(f"      实际范围: [{start_index}, {end_index})")
        print(f"      实际处理数量: {len(image_files)}")
    else:
        print(f"✗ 无效的路径类型")
        return

    if len(image_files) == 0:
        print("✗ 当前切片范围内没有可处理的图像文件")
        return

    mask_input = None
    mask_dir = None

    needs_mask = any(model.requires_mask for model in [registry.get_model(name) for name in registry.list_models()])

    if needs_mask:
        mask_input = paths_config['data'].get('mask_input', '')

        print(f">>> 加载掩码路径")
        print(f"    掩码路径: {mask_input if mask_input else '(未配置)'}")

        if mask_input and os.path.exists(mask_input):
            mask_input_path = Path(mask_input)
            if mask_input_path.is_dir():
                mask_dir = mask_input_path
                print(f"✓ 检测到掩码目录")
            else:
                print(f"✓ 检测到单个掩码文件")
        elif mask_input:
            print(f"⚠️  掩码路径不存在: {mask_input}")
            print("   需要掩码的模型将无法运行")
        else:
            print(f"⚠️  未配置掩码路径，需要掩码的模型将无法运行")

    print()

    print("=" * 70)
    print(f"开始批量处理 ({len(image_files)} 个图像)")
    print("=" * 70)
    print()

    print(">>> 步骤 1/3: 加载所有图像和掩码")
    image_data = []

    for idx, image_file in enumerate(image_files, 1):
        print(f"  [{idx}/{len(image_files)}] 加载 {image_file.name}...", end=" ")

        try:
            image = image_processor.load_image(str(image_file))

            mask = None
            if mask_dir:
                mask_file = find_corresponding_mask(image_file, mask_dir)
                if mask_file:
                    mask = image_processor.load_mask(str(mask_file))
            elif mask_input and Path(mask_input).is_file() and len(image_files) == 1:
                mask = image_processor.load_mask(mask_input)

            image_data.append((image_file, image, mask))
            print("✓")

        except Exception as e:
            print(f"✗ 失败: {e}")

    print(f"✓ 成功加载 {len(image_data)} 个图像\n")

    if len(image_data) == 0:
        print("✗ 没有成功加载的图像，退出")
        return

    print(">>> 步骤 2/3: 使用每个模型对所有图像进行推理")

    all_predictions = [{} for _ in range(len(image_data))]

    for model_name in registry.list_models():
        model = registry.get_model(model_name)

        print(f"\n  模型: {model.model_name}")
        print(f"  {'='*65}")

        batch_images = []
        batch_masks = []
        batch_indices = []

        for idx, (image_file, image, mask) in enumerate(image_data):
            if model.requires_mask and mask is None:
                print(f"    [{idx+1}/{len(image_data)}] {image_file.name}... 跳过 (缺少掩码)")
                continue

            batch_images.append(image)
            batch_masks.append(mask)
            batch_indices.append(idx)

        if len(batch_images) == 0:
            print(f"  ⚠️  没有可用的图像（可能缺少掩码），跳过此模型\n")
            continue

        try:
            is_autogluon = (
                hasattr(model, 'predict_batch') and
                (model.model_name.startswith("AutoGluon_") or
                 model.model_name == "AutoGluon_PyRadiomics" or
                 isinstance(model, AutoGluonRadiomicsModel))
            )

            if is_autogluon:
                results = model.predict_batch(batch_images, batch_masks, show_progress=True)
                for r in results:
                    if r is not None:
                        try:
                            maybe_apply_calibration_map(r, registry.calibration_map)
                        except Exception:
                            pass
            else:
                results = []
                for idx, (image, mask) in enumerate(zip(batch_images, batch_masks)):
                    image_file = image_data[batch_indices[idx]][0]
                    print(f"    [{idx+1}/{len(batch_images)}] {image_file.name}...", end=" ")

                    try:
                        result = model.predict(image, mask)
                        try:
                            maybe_apply_calibration_map(
                                result, registry.calibration_map
                            )
                        except Exception:
                            pass
                        results.append(result)
                        print(f"✓ {result.top_class} ({result.top_confidence:.4f})")
                    except Exception as e:
                        print(f"✗ 失败: {e}")
                        results.append(None)

            for result, orig_idx in zip(results, batch_indices):
                if result is not None:
                    all_predictions[orig_idx][model_name] = result

            print(f"  ✓ {model.model_name} 完成所有推理\n")

        except Exception as e:
            print(f"  ✗ {model.model_name} 批量推理失败: {e}")
            import traceback
            traceback.print_exc()
            print()

    valid_data = []
    for idx, (image_file, image, mask) in enumerate(image_data):
        if len(all_predictions[idx]) > 0:
            valid_data.append((idx, image_file, all_predictions[idx]))

    if len(valid_data) == 0:
        print("✗ 没有图像有成功的推理结果，退出")
        return

    print(f"✓ 共 {len(valid_data)} 个图像有成功的推理结果\n")

    print(">>> 步骤 3/3: 决策与结果汇总")
    print("=" * 70)

    output_dir = Path(paths_config['output'].get('output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    decision_threshold = float(paths_config.get("agent", {}).get("decision_threshold", 0.5))
    labels_for_results: Optional[dict[str, int]] = None
    if resolved_label_path is not None and resolved_label_path.exists():
        try:
            labels_for_results = _load_labels(resolved_label_path)
        except Exception as e:
            print(f"⚠️ 标签文件加载失败，结果中将不写入 ground_truth_label: {e}")

    all_results = []

    if enable_agent and agent is not None:
        batch_predictions = []
        for idx, image_file, predictions_dict in valid_data:
            batch_predictions.append({
                "image_file": str(image_file),
                "image_name": image_file.name,
                "predictions": list(predictions_dict.values())
            })

        try:
            input_device_info = paths_config['data'].get('device_info', None)
            if input_device_info:
                print(f"\n输入数据设备信息: {', '.join(input_device_info)}")
            input_data_info = paths_config.get('data', {})

            print(f"\n正在处理 {len(batch_predictions)} 个图像的综合决策（使用 Agent）...\n")
            batch_decisions = agent.select_best_model_batch(
                batch_predictions,
                input_device_info=input_device_info,
                input_data_info=input_data_info,
                incremental_save_path=str(output_file),
            )

            print("✓ Agent 批量决策完成!\n")

            print("=" * 70)
            print("决策结果（Agent 模式）")
            print("=" * 70)

            for i, (idx, image_file, predictions_dict) in enumerate(valid_data):
                decision = batch_decisions[i]

                print(f"\n[{i+1}/{len(valid_data)}] {image_file.name}")
                print(f"  选择模型: {decision.selected_model}")
                print(f"  预测类别: {decision.selected_class}")
                print(f"  置信度: {decision.confidence:.4f}")
                print(f"  理由: {decision.reasoning}")

                result_dict = _build_result_dict(
                    image_file=image_file,
                    selected_model=decision.selected_model,
                    predicted_class=decision.selected_class,
                    confidence=float(decision.confidence),
                    reasoning=decision.reasoning,
                    predictions=list(predictions_dict.values()),
                    labels=labels_for_results,
                    decision_threshold=decision_threshold,
                )
                all_results.append(result_dict)

        except Exception as e:
            print(f"\n✗ Agent 批量决策失败: {e}")
            import traceback
            traceback.print_exc()

            print("\n尝试回退到单张决策模式（仍使用 Agent）...")

            for idx, image_file, predictions_dict in valid_data:
                try:
                    predictions = list(predictions_dict.values())
                    decision = agent.select_best_model(
                        predictions,
                        input_device_info=paths_config.get('data', {}).get('device_info', None),
                        input_data_info=paths_config.get('data', {})
                    )

                    result_dict = _build_result_dict(
                        image_file=image_file,
                        selected_model=decision.selected_model,
                        predicted_class=decision.selected_class,
                        confidence=float(decision.confidence),
                        reasoning=decision.reasoning,
                        predictions=predictions,
                        labels=labels_for_results,
                        decision_threshold=decision_threshold,
                    )
                    all_results.append(result_dict)

                except Exception as e2:
                    print(f"✗ 单张决策也失败: {image_file.name}: {e2}")
    else:
        print("Agent LLM 已关闭，使用 top-k soft voting 策略进行综合决策。\n")

        print("=" * 70)
        print("决策结果（Soft Voting 模式）")
        print("=" * 70)

        for i, (idx, image_file, predictions_dict) in enumerate(valid_data):
            predictions = list(predictions_dict.values())
            sorted_preds = sorted(
                predictions, key=lambda p: p.top_confidence, reverse=True
            )
            k = min(top_k, len(sorted_preds))
            subset = sorted_preds[:k]

            avg_class_probs = _average_class_probabilities(subset)
            selected_class, best_confidence = _winning_class_from_avg_probs(avg_class_probs)

            selected_model_name = "soft_voting_topk_ensemble"

            reasoning = (
                f"未使用 Agent，按 top_confidence 取前 {k} 个模型后对类别概率取均值，"
                f"类别 '{selected_class}' 的平均概率最高（{best_confidence:.4f}）。"
            )

            print(f"\n[{i+1}/{len(valid_data)}] {image_file.name}")
            print(f"  选择模型: {selected_model_name}")
            print(f"  预测类别: {selected_class}")
            print(f"  置信度: {best_confidence:.4f}")
            print(f"  理由: {reasoning}")

            result_dict = _build_result_dict(
                image_file=image_file,
                selected_model=selected_model_name,
                predicted_class=selected_class,
                confidence=float(best_confidence),
                reasoning=reasoning,
                predictions=predictions,
                labels=labels_for_results,
                decision_threshold=decision_threshold,
            )
            all_results.append(result_dict)

    if len(all_results) > 0:
        print("\n" + "=" * 70)
        print("保存结果")
        print("=" * 70)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"✓ 结果已保存到: {output_file}")
        print(f"  共处理 {len(all_results)} 个图像")

        print("\n【统计信息】")
        model_counts = {}
        class_counts = {}

        for result in all_results:
            model = result["selected_model"]
            pred_class = result["predicted_class"]

            model_counts[model] = model_counts.get(model, 0) + 1
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        print(f"\n模型选择统计:")
        for model, count in model_counts.items():
            print(f"  {model}: {count} 次 ({count/len(all_results)*100:.1f}%)")

        print(f"\n类别预测统计:")
        for pred_class, count in class_counts.items():
            print(f"  {pred_class}: {count} 次 ({count/len(all_results)*100:.1f}%)")

        decision_threshold = float(paths_config.get("agent", {}).get("decision_threshold", 0.5))

        def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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

        def _compute_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
            y_pred_bin = (y_prob >= threshold).astype(np.int32)
            acc = float(np.mean((y_pred_bin == y_true).astype(np.float64)))
            prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
            rec = float(recall_score(y_true, y_pred_bin, zero_division=0))
            f1v = float(f1_score(y_true, y_pred_bin, zero_division=0))

            tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
            fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

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

            ece = _ece_binary(y_true, y_prob, n_bins=10)
            return {
                "auroc": auroc,
                "auprc": auprc,
                "acc": acc,
                "prec": prec,
                "recall": rec,
                "f1": f1v,
                "specificity": specificity,
                "ece": ece,
            }

        def _compute_bootstrap_ci95(
            y_true: np.ndarray,
            y_prob: np.ndarray,
            threshold: float,
            n_boot: int = 2000,
            seed: int = 0,
        ) -> dict[str, dict[str, Optional[float]]]:
            rng = np.random.default_rng(seed)
            n = len(y_true)
            metric_keys = ["auroc", "auprc", "acc", "prec", "recall", "f1", "specificity", "ece"]
            samples = {k: [] for k in metric_keys}

            for _ in range(max(1, n_boot)):
                idx = rng.integers(0, n, n)
                yt = y_true[idx]
                yp = y_prob[idx]
                m = _compute_point_metrics(yt, yp, threshold)
                for k in metric_keys:
                    samples[k].append(float(m[k]) if not np.isnan(m[k]) else np.nan)

            out: dict[str, dict[str, Optional[float]]] = {}
            for k in metric_keys:
                arr = np.asarray(samples[k], dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    out[k] = {"mean": None, "ci95_lower": None, "ci95_upper": None}
                    continue
                out[k] = {
                    "mean": float(np.mean(valid)),
                    "ci95_lower": float(np.percentile(valid, 2.5)),
                    "ci95_upper": float(np.percentile(valid, 97.5)),
                }
            return out

        try:
            labels = None
            label_path = resolved_label_path
            if label_path is not None and label_path.exists():
                labels = _load_labels(label_path)

            y_true_list: list[int] = []
            y_prob_list: list[float] = []
            for r in all_results:
                gt = r.get("ground_truth_label", None)
                if gt is None and labels is not None:
                    image_name = r.get("image_name") or ""
                    gt = _label_lookup(labels, image_name)
                if gt is None:
                    continue

                p_malignant = r.get("malignant_probability", None)
                if p_malignant is None:
                    pred_class = str(r.get("predicted_class", "")).strip()
                    conf = float(r.get("confidence", 0.5))
                    p_malignant = _infer_malignant_probability(pred_class, conf)

                y_true_list.append(int(gt))
                y_prob_list.append(float(p_malignant))

            y_true = np.asarray(y_true_list, dtype=np.int32)
            y_prob = np.asarray(y_prob_list, dtype=np.float64)
            n_eval = int(y_true.shape[0])

            if n_eval == 0:
                print("\n【平均分类指标】跳过：results 中缺少可用于评估的 ground_truth_label/malignant_probability，且无法通过 labels 对齐")
            else:
                n_boot = int(paths_config.get("agent", {}).get("metrics_n_boot", 2000))
                boot_seed = int(paths_config.get("agent", {}).get("metrics_bootstrap_seed", 0))
                point = _compute_point_metrics(y_true, y_prob, decision_threshold)
                ci95 = _compute_bootstrap_ci95(
                    y_true=y_true,
                    y_prob=y_prob,
                    threshold=decision_threshold,
                    n_boot=n_boot,
                    seed=boot_seed,
                )

                print("\n【平均分类指标】")
                print(f"  样本数: {n_eval}")
                print(f"  Bootstrap: n_boot={n_boot}, seed={boot_seed}")
                pretty_names = [
                    ("AUROC", "auroc"),
                    ("AUPRC", "auprc"),
                    ("Acc", "acc"),
                    ("Prec", "prec"),
                    ("Recall", "recall"),
                    ("F1", "f1"),
                    ("Specificity", "specificity"),
                    ("ECE", "ece"),
                ]
                for display, key in pretty_names:
                    v = point[key]
                    c = ci95[key]
                    if v is None or np.isnan(v) or c["mean"] is None:
                        print(f"  {display}: N/A")
                    else:
                        print(
                            f"  {display}: {float(v):.4f}  "
                            f"(mean={float(c['mean']):.4f}, CI95=[{float(c['ci95_lower']):.4f}, {float(c['ci95_upper']):.4f}])"
                        )

                metrics_out = {
                    "label_path": None if label_path is None else str(label_path),
                    "n_samples": n_eval,
                    "threshold": decision_threshold,
                    "n_boot": n_boot,
                    "metrics": {k: (None if np.isnan(v) else round(float(v), 6)) for k, v in point.items()},
                    "metrics_ci95": {
                        k: {
                            "mean": None if v["mean"] is None else round(float(v["mean"]), 6),
                            "ci95_lower": None if v["ci95_lower"] is None else round(float(v["ci95_lower"]), 6),
                            "ci95_upper": None if v["ci95_upper"] is None else round(float(v["ci95_upper"]), 6),
                        }
                        for k, v in ci95.items()
                    },
                }
                metrics_out_path = output_dir / f"agent_metrics_{timestamp}.json"
                with open(metrics_out_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_out, f, ensure_ascii=False, indent=2)
                print(f"  指标已保存: {metrics_out_path}")
        except Exception as e:
            print(f"\n【平均分类指标】计算失败（已跳过）：{e}")

    print("\n" + "=" * 70)
    print("运行完成")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thyroid classification pipeline with configurable YAML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
