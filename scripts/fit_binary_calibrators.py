"""
离线拟合二分类（甲状腺良/恶性）校准器：在带标签的验证集上逐模型拟合并保存 JSON。

用法（在项目根目录执行）:
  python scripts/fit_binary_calibrators.py --config config/config_Superimposed_dataset.yaml --method platt
  python scripts/fit_binary_calibrators.py --help

输出: 默认写入 calibration/artifacts/<model_name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 项目根加入路径
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calibration.binary_calibration import (  # noqa: E402
    build_calibrator_artifact,
    extract_p_malignant,
    fit_binary_calibrator,
    save_calibrator_json,
)
from main import (  # noqa: E402
    derive_training_data_devices,
    find_corresponding_mask,
    get_image_files,
    load_config,
    resolve_label_path,
)
from models.autogluon_radiomics_model import AutoGluonRadiomicsModel  # noqa: E402
from models.dino_unet_model import DINOUNetModel  # noqa: E402
from models.model_registry import ModelRegistry  # noqa: E402
from utils.image_processor import ImageProcessor  # noqa: E402

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def _dino_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_label_mapping(label_path: Path) -> Dict[str, int]:
    """
    支持:
    - [{ "filename": "a.jpg", "malignancy": 0|1 }, ...]
    - { "a.jpg": 0, ... }  （值为 0/1 或 bool）
    """
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fn = item.get("filename")
            if fn is None:
                continue
            mal = item.get("malignancy", item.get("label"))
            if mal is None:
                continue
            out[str(fn)] = int(mal)
        return out
    if isinstance(data, dict):
        for k, v in data.items():
            out[str(k)] = int(v)
        return out
    raise ValueError(f"Unsupported label format in {label_path}")


def build_registry_from_config(config: Dict[str, Any]) -> ModelRegistry:
    """与 demo_thyroid_models 中注册逻辑一致，便于复用同一套 yaml。"""
    registry = ModelRegistry()
    paths_config = config

    dino_models_config = paths_config.get("dino_unet", {}).get("models", [])
    for idx, model_config in enumerate(dino_models_config, 1):
        model_name = model_config.get("name", f"dino_unet_{idx}")
        model_path = model_config["model_path"]
        use_tirads = model_config.get("use_tirads", False)
        if use_tirads:
            print(f"  [跳过] DINO TI-RADS 非二分类，不拟合校准: {model_name}")
            continue
        if not os.path.exists(model_path):
            print(f"  [跳过] DINO 权重不存在: {model_path}")
            continue
        try:
            dino_model = DINOUNetModel(
                model_path=model_path,
                device=_dino_device(),
                use_tirads=use_tirads,
            )
            dino_model.load_model()
            if "name" in model_config:
                dino_model.model_name = f"DINO_UNet_{model_name}"
            dataset_info = model_config.get("dataset_info", None)
            if dataset_info:
                dino_model.dataset_info = dataset_info
            training_data_devices = model_config.get("training_data_devices", None)
            if training_data_devices is None or len(training_data_devices) == 0:
                base_datasets_info = paths_config.get("base_datasets_info", None)
                training_data_devices = derive_training_data_devices(
                    dataset_info, base_datasets_info
                )
            dino_model.training_data_devices = training_data_devices
            validation_metrics = model_config.get("validation_metrics", None)
            if validation_metrics:
                dino_model.validation_metrics = validation_metrics
            base_dataset_performance = model_config.get("base_dataset_performance", None)
            if base_dataset_performance:
                dino_model.base_dataset_performance = base_dataset_performance
            registry.register_model(dino_model)
        except Exception as e:
            print(f"  [失败] DINO {model_name}: {e}")

    autogluon_models_config = paths_config.get("autogluon", {}).get("models", [])
    for idx, model_config in enumerate(autogluon_models_config, 1):
        model_name = model_config.get("name", f"autogluon_{idx}")
        model_dir = model_config["model_dir"]
        if not os.path.exists(model_dir):
            print(f"  [跳过] AutoGluon 目录不存在: {model_dir}")
            continue
        try:
            autogluon_model = AutoGluonRadiomicsModel(model_dir=model_dir)
            autogluon_model.load_model()
            if "name" in model_config:
                autogluon_model.model_name = f"AutoGluon_{model_name}"
            dataset_info = model_config.get("dataset_info", None)
            if dataset_info:
                autogluon_model.dataset_info = dataset_info
            training_data_devices = model_config.get("training_data_devices", None)
            if training_data_devices is None or len(training_data_devices) == 0:
                base_datasets_info = paths_config.get("base_datasets_info", None)
                training_data_devices = derive_training_data_devices(
                    dataset_info, base_datasets_info
                )
            autogluon_model.training_data_devices = training_data_devices
            validation_metrics = model_config.get("validation_metrics", None)
            if validation_metrics:
                autogluon_model.validation_metrics = validation_metrics
            base_dataset_performance = model_config.get("base_dataset_performance", None)
            if base_dataset_performance:
                autogluon_model.base_dataset_performance = base_dataset_performance
            registry.register_model(autogluon_model)
        except Exception as e:
            print(f"  [失败] AutoGluon {model_name}: {e}")

    return registry


def collect_pairs(
    registry: ModelRegistry,
    image_dir: Path,
    mask_dir: Optional[Path],
    label_map: Dict[str, int],
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
    """
    按 model_name 收集 (p_malignant, y)。仅处理 label_map 中存在的文件名。
    """
    image_files = get_image_files(image_dir)
    if not image_files:
        raise RuntimeError(f"目录下无图像: {image_dir}")

    processor = ImageProcessor()
    by_model: Dict[str, List[float]] = {name: [] for name in registry.list_models()}
    by_y: Dict[str, List[int]] = {name: [] for name in registry.list_models()}

    used = 0
    for img_path in image_files:
        name = img_path.name
        if name not in label_map:
            continue
        y = int(label_map[name])
        mask = None
        if mask_dir is not None:
            mp = find_corresponding_mask(img_path, mask_dir)
            if mp is not None:
                mask = processor.load_mask(mp)

        try:
            image = processor.load_image(img_path)
        except Exception as e:
            print(f"  [跳过] 读图失败 {name}: {e}")
            continue

        for mname, model in registry.models.items():
            try:
                if getattr(model, "requires_mask", False) and mask is None:
                    continue
                model.validate_inputs(image, mask)
                out = model.predict(image, mask)
            except Exception as e:
                print(f"  [跳过] {mname} 预测失败 {name}: {e}")
                continue
            pm = extract_p_malignant(out)
            if pm is None:
                print(f"  [跳过] {mname} 无恶性概率键: {name}")
                continue
            by_model[mname].append(pm)
            by_y[mname].append(y)

        used += 1
        if max_samples is not None and used >= max_samples:
            break

    return by_model, by_y


def main() -> None:
    parser = argparse.ArgumentParser(description="离线拟合二分类校准器并保存 JSON")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_Superimposed_dataset.yaml",
        help="YAML 配置（与 demo 相同）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("temperature", "platt", "isotonic"),
        default="platt",
        help="校准方法",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="calibration/artifacts",
        help="校准器 JSON 输出目录",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help="覆盖 config 中的 data.label_file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="覆盖 config 中的 data.image_input（仅目录）",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="覆盖 config 中的 data.mask_input；可为空字符串表示禁用掩码",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多使用多少张有标签的图像（调试用）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    config = load_config(str(config_path))
    if not config:
        raise SystemExit("配置加载失败")

    data_cfg = config.get("data", {}) or {}
    image_input = args.image_dir or data_cfg.get("image_input")
    if not image_input:
        raise SystemExit("未配置 data.image_input")
    image_path = Path(image_input)
    if not image_input.startswith("/") and not image_path.is_absolute():
        image_path = _ROOT / image_path
    if not image_path.is_dir():
        raise SystemExit(f"图像目录不存在或不是目录: {image_path}")

    mask_dir: Optional[Path] = None
    if args.mask_dir is not None:
        if str(args.mask_dir).strip() == "":
            mask_dir = None
        else:
            mp = Path(args.mask_dir)
            if not mp.is_absolute():
                mp = _ROOT / mp
            mask_dir = mp if mp.is_dir() else None
    else:
        mi = data_cfg.get("mask_input")
        if mi and str(mi).strip().lower() != "null":
            mp = Path(mi)
            if not mp.is_absolute():
                mp = _ROOT / mp
            mask_dir = mp if mp.is_dir() else None

    label_path: Optional[Path] = None
    if args.label_file:
        label_path = Path(args.label_file)
        if not label_path.is_absolute():
            label_path = _ROOT / label_path
    else:
        resolved, src = resolve_label_path(config)
        if resolved is None:
            raise SystemExit(
                "未找到标签文件。请在 config 中设置 data.label_file 或使用 --label-file"
            )
        label_path = resolved
        print(f"标签文件 ({src}): {label_path}")

    label_map = load_label_mapping(label_path)
    print(f"已加载 {len(label_map)} 条标签")

    print("注册模型...")
    registry = build_registry_from_config(config)
    if len(registry) == 0:
        raise SystemExit("没有可用的模型")

    print(f"收集预测 ({image_path}) ...")
    by_p, by_y = collect_pairs(
        registry,
        image_path,
        mask_dir,
        label_map,
        max_samples=args.max_samples,
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    method = args.method
    for mname in registry.list_models():
        p_raw = np.asarray(by_p[mname], dtype=np.float64)
        y = np.asarray(by_y[mname], dtype=np.float64)
        if len(p_raw) < 2:
            print(f"  [跳过] {mname}: 有效样本不足 ({len(p_raw)})")
            continue
        if np.unique(y).size < 2:
            print(f"  [跳过] {mname}: 标签仅含单一类别")
            continue
        try:
            fit = fit_binary_calibrator(p_raw, y, method=method)  # type: ignore[arg-type]
        except Exception as e:
            print(f"  [失败] {mname} 拟合: {e}")
            continue
        artifact = build_calibrator_artifact(mname, fit)
        safe = mname.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe}.json"
        save_calibrator_json(artifact, out_path)
        print(
            f"  ✓ {mname} -> {out_path} | "
            f"NLL {fit.metrics_raw['nll']:.4f} -> {fit.metrics_cal['nll']:.4f} | "
            f"ECE {fit.metrics_raw['ece']:.4f} -> {fit.metrics_cal['ece']:.4f}"
        )

    print(f"完成。校准器目录: {out_dir}")


if __name__ == "__main__":
    main()
