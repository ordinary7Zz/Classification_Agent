"""
Gemini/GLM-based agent for selecting the best model prediction
Uses an OpenAI-compatible API endpoint (e.g., Zhipu GLM)
"""

import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI

from models.base_model import ModelOutput


@dataclass
class AgentDecision:
    """
    Agent's decision on which model has the best prediction
    """
    selected_model: str
    selected_class: str
    confidence: float
    reasoning: str
    all_predictions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'selected_model': self.selected_model,
            'selected_class': self.selected_class,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'all_predictions': self.all_predictions
        }


class GeminiAgent:
    """
    GLM/Qwen-powered agent for selecting the best classification result
    Uses an OpenAI-compatible API endpoint (e.g., Zhipu GLM)
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        base_datasets_info: Optional[Dict[str, Any]] = None,
        max_batch_size: int = 10,
        selection_mode: str = "deterministic",
        ):
        """
        Initialize the agent via an OpenAI-compatible API (e.g., Zhipu GLM)
        
        Args:
            api_key: OpenAI-compatible API key (e.g., GLM_API_KEY)
            model_name: Model name on the provider (e.g., "glm-4-air")
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            base_datasets_info: Optional dict containing base datasets device mapping
            max_batch_size: Maximum number of images to process in a single API call (default: 10)
            selection_mode: Selection strategy mode (kept for backward compatibility)
        """
        # Allow passing api_key directly or via environment variable GLM_API_KEY
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError("GLM_API_KEY is not set and no api_key was provided.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_datasets_info = base_datasets_info or {}
        self.max_batch_size = max_batch_size
        self.selection_mode = selection_mode
        # OpenAI-compatible client (default base_url points to Zhipu GLM)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        
        # System prompt（短版+结构化约束，控制 token 同时提升一致性）
        self.system_prompt = """你是甲状腺超声多模型预测整合专家，从若干模型输出中选最可信的一项。

【设备】设备决定成像风格；训练数据覆盖输入同款/同品牌者更可信。GE(Logiq E9/S7 等)与 Hitachi(ARIETTA 等)各系内部风格近；其余品牌与上述有差异。Heterogeneous=多设备混合。输入设备未知则忽略此项。

【字段】主置信度优先 metadata.classification_uncertainty.top_confidence_calibrated，否则 top_confidence_raw 或 top_confidence。entropy(越大越不确定)、margin_top2(越大越稳) 在同路径下。consistency_metrics：num_models_same_class、total_models、vote_entropy。

【决策序】1)主置信度 2)已知输入设备则设备匹配 3)主置信度差<0.05 时比 validation_metrics.on_training_dataset 的 acc/AUC/F1 4)仍接近则 entropy 更低、margin_top2 更高 5)能推断 TN3K/ThyroidXL/TN5K/CineClip 则看 base_dataset_performance，否则 dataset_size 更大优先 6)差<0.05 结合投票与 num_models_same_class；类别冲突时若有单模型主置信度>0.95 可优先 7)仅当差<0.02 再考虑模型结构差异。

【输出】只输出纯 JSON（无 Markdown/思考/代码块），首尾为 { }，字段：
- selected_model, selected_class, confidence
- runner_up_model, runner_up_confidence, delta_confidence
- triggered_rules（如 ["R1","R3"]）
- reasoning

【一致性】delta_confidence=confidence-runner_up_confidence；比较词须与数值一致（delta>=0.05 才能写“显著高于/远高于”，否则写“高于/接近”）。"""
    
    def format_predictions(self, predictions: List[ModelOutput]) -> str:
        """
        Format model predictions for the agent
        
        Args:
            predictions: List of ModelOutput from different models
            
        Returns:
            Formatted string representation of predictions
        """
        formatted = "# Model Predictions Summary\n\n"
        
        for idx, pred in enumerate(predictions, 1):
            formatted += f"## Model {idx}: {pred.model_name}\n"
            formatted += f"- **Top Prediction**: {pred.top_class}\n"
            formatted += f"- **Confidence**: {pred.top_confidence:.4f}\n"
            formatted += f"- **Requires Mask**: {pred.requires_mask}\n"
            
            # Add training data device information if available
            if pred.metadata and 'training_data_devices' in pred.metadata:
                devices = pred.metadata['training_data_devices']
                if devices:
                    formatted += f"- **Training Data Devices**: {', '.join(devices)}\n"
                else:
                    formatted += f"- **Training Data Devices**: Unknown\n"
            
            # Add dataset info if available
            if pred.metadata and 'dataset_info' in pred.metadata:
                ds_info = pred.metadata['dataset_info']
                formatted += f"- **Training Dataset**: {ds_info.get('training_dataset', 'Unknown')}\n"
                if 'base_datasets' in ds_info:
                    formatted += f"- **Base Datasets**: {', '.join(ds_info['base_datasets'])}\n"
                if 'dataset_size' in ds_info:
                    formatted += f"- **Dataset Size**: {ds_info['dataset_size']}\n"
            
            # Add validation metrics if available
            if pred.metadata and 'validation_metrics' in pred.metadata:
                val_metrics = pred.metadata['validation_metrics']
                if 'on_training_dataset' in val_metrics:
                    on_train = val_metrics['on_training_dataset']
                    formatted += f"- **Validation Metrics (on training dataset)**: "
                    metrics_str = []
                    if 'accuracy' in on_train:
                        metrics_str.append(f"Acc={on_train['accuracy']:.3f}")
                    if 'auc' in on_train:
                        metrics_str.append(f"AUC={on_train['auc']:.3f}")
                    if 'f1_score' in on_train:
                        metrics_str.append(f"F1={on_train['f1_score']:.3f}")
                    formatted += ", ".join(metrics_str) + "\n"
            
            # Add base dataset performance if available
            if pred.metadata and 'base_dataset_performance' in pred.metadata:
                base_perf = pred.metadata['base_dataset_performance']
                formatted += f"- **Base Dataset Performance**: Available (TN3K, ThyroidXL, TN5K, CineClip)\n"
            
            formatted += f"- **All Predictions**:\n"
            
            # Sort predictions by confidence
            sorted_preds = sorted(
                pred.predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for class_name, conf in sorted_preds[:5]:  # Top 5
                formatted += f"  - {class_name}: {conf:.4f}\n"
            
            formatted += "\n"
        
        return formatted
    
    def format_predictions_json(self, predictions: List[ModelOutput]) -> str:
        """
        Format model predictions as JSON for the agent
        
        Args:
            predictions: List of ModelOutput from different models
            
        Returns:
            JSON string representation of predictions
        """
        predictions_data = self._build_compact_prediction_dicts(predictions)
        data = {"num_models": len(predictions), "predictions": predictions_data}
        return json.dumps(data, ensure_ascii=False)  # 不缩进以减小 prompt 体积

    def _build_compact_prediction_dicts(self, predictions: List[ModelOutput]) -> List[Dict[str, Any]]:
        """
        Build compact prediction payload for LLM input.
        Keeps decision-critical fields while dropping large, non-essential metadata.
        """
        import math

        votes_per_class: Dict[str, int] = {}
        for pred in predictions:
            votes_per_class[pred.top_class] = votes_per_class.get(pred.top_class, 0) + 1

        total_models = len(predictions)
        vote_entropy = 0.0
        if total_models > 0:
            probs = [count / total_models for count in votes_per_class.values()]
            vote_entropy = -sum(p * math.log(p + 1e-12, 2) for p in probs if p > 0)

        compact_predictions: List[Dict[str, Any]] = []
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
            top_conf_calibrated = cu.get("top_confidence_calibrated")

            on_train = (metadata.get("validation_metrics", {}) or {}).get("on_training_dataset", {}) or {}
            dataset_info = metadata.get("dataset_info", {}) or {}
            base_perf = metadata.get("base_dataset_performance", {}) or {}
            train_devices = metadata.get("training_data_devices") or []

            compact_predictions.append({
                "model_name": pred_dict.get("model_name"),
                "top_class": pred_dict.get("top_class"),
                "top_confidence": pred_dict.get("top_confidence"),
                "top2_predictions": top_k_predictions,
                "metadata": {
                    "classification_uncertainty": {
                        **({"top_confidence_calibrated": top_conf_calibrated} if top_conf_calibrated is not None else {}),
                        "top_confidence_raw": pred_dict.get("top_confidence"),
                        "entropy": entropy,
                        "margin_top2": margin_top2,
                    },
                    "consistency_metrics": {
                        "num_models_same_class": votes_per_class.get(pred_dict.get("top_class"), 0),
                        "total_models": total_models,
                        "vote_entropy": vote_entropy,
                    },
                    "training_data_devices": train_devices,
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

        return compact_predictions
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text, handling markdown code blocks, thinking process, and nested structures
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Remove common prefixes from thinking process
        if text.startswith("*Thinking"):
            # Skip to after thinking section
            parts = text.split("\n\n")
            for i, part in enumerate(parts):
                if '{' in part:
                    text = '\n\n'.join(parts[i:])
                    break
        
        # First, try to extract from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            # Find the part with JSON
            for part in parts:
                if '{' in part and '}' in part:
                    text = part.strip()
                    break
        
        # Try to find JSON object by finding the first { and matching closing }
        # This handles nested JSON structures
        start_idx = text.find('{')
        if start_idx == -1:
            return text
        
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the matching closing brace
                        json_str = text[start_idx:i+1]
                        # Clean control characters that might cause JSON parsing issues
                        import re
                        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                        return json_str
        
        # If we couldn't find a complete JSON object, return what we have
        json_str = text[start_idx:]
        import re
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        return json_str

    @staticmethod
    def _predictions_top_class_unanimous(predictions: List[ModelOutput]) -> bool:
        """所有模型 top_class 相同时为 True（含仅 1 个模型）。"""
        if not predictions:
            return False
        first = predictions[0].top_class
        return all(p.top_class == first for p in predictions)

    def _decision_unanimous_max_confidence(self, predictions: List[ModelOutput]) -> AgentDecision:
        """各模型类别一致时：不调用大模型，取 top_confidence 最高者。"""
        best = max(predictions, key=lambda p: p.top_confidence)
        cls = best.top_class
        reasoning = (
            f"所有模型均预测为「{cls}」，决策一致，未调用大模型；"
            f"选取 top_confidence 最高的模型 {best.model_name}（{best.top_confidence:.4f}）。"
        )
        return AgentDecision(
            selected_model=best.model_name,
            selected_class=cls,
            confidence=float(best.top_confidence),
            reasoning=reasoning,
            all_predictions=[pred.to_dict() for pred in predictions],
        )

    @staticmethod
    def _post_check_structured_fields(decision_data: Dict[str, Any], predictions: List[ModelOutput]) -> Dict[str, Any]:
        """
        Fill/repair lightweight structured fields to reduce numeric contradictions.
        """
        conf_map = {p.model_name: float(p.top_confidence) for p in predictions}
        selected = decision_data.get("selected_model")
        if selected in conf_map:
            decision_data["confidence"] = float(conf_map[selected])

        sorted_preds = sorted(predictions, key=lambda p: p.top_confidence, reverse=True)
        runner = None
        for p in sorted_preds:
            if p.model_name != selected:
                runner = p
                break
        if runner is not None:
            decision_data.setdefault("runner_up_model", runner.model_name)
            decision_data.setdefault("runner_up_confidence", float(runner.top_confidence))
            decision_data["delta_confidence"] = float(decision_data["confidence"]) - float(decision_data["runner_up_confidence"])
        else:
            decision_data.setdefault("runner_up_model", "")
            decision_data.setdefault("runner_up_confidence", 0.0)
            decision_data.setdefault("delta_confidence", 0.0)

        if "triggered_rules" not in decision_data or not isinstance(decision_data.get("triggered_rules"), list):
            decision_data["triggered_rules"] = []
        return decision_data

    @staticmethod
    def _format_input_data_info_text(
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build input data context text for prompt.
        Rule: null / None / empty means unknown.
        """
        lines: List[str] = []
        data_info = input_data_info or {}

        def _is_unknown(v: Any) -> bool:
            if v is None:
                return True
            if isinstance(v, str):
                return v.strip() == "" or v.strip().lower() == "null"
            if isinstance(v, (list, tuple, set, dict)):
                return len(v) == 0
            return False

        # Device info: explicit arg has higher priority, then data.device_info
        device_info = input_device_info
        if _is_unknown(device_info):
            device_info = data_info.get("device_info")

        if _is_unknown(device_info):
            lines.append("- device_info: 未知")
        elif isinstance(device_info, list):
            lines.append(f"- device_info: {', '.join(str(x) for x in device_info)}")
        else:
            lines.append(f"- device_info: {device_info}")

        for key in ["image_input", "mask_input", "label_file"]:
            value = data_info.get(key)
            if _is_unknown(value):
                lines.append(f"- {key}: 未知")
            else:
                lines.append(f"- {key}: {value}")

        return "\n输入数据上下文（data；null=未知）:\n" + "\n".join(lines) + "\n"

    def select_best_model(
        self,
        predictions: List[ModelOutput],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """
        Use Gemini agent to select the best model prediction
        
        Args:
            predictions: List of ModelOutput from different models
            use_json_format: Whether to format predictions as JSON
            input_device_info: Optional list of device information for input data (e.g., ["GE", "Siemens"])
                             If None, device information is unknown
            input_data_info: Optional full data context from config.data. null/None fields are treated as unknown.
            
        Returns:
            AgentDecision with the selected model and reasoning

        若所有模型的 top_class 一致，则不调用大模型，直接取 top_confidence 最高的模型。
        """
        if not predictions:
            raise ValueError("No predictions provided")

        if self._predictions_top_class_unanimous(predictions):
            return self._decision_unanimous_max_confidence(predictions)

        # Format predictions
        if use_json_format:
            formatted_preds = self.format_predictions_json(predictions)
        else:
            formatted_preds = self.format_predictions(predictions)
        
        data_info_text = self._format_input_data_info_text(
            input_device_info=input_device_info,
            input_data_info=input_data_info
        )

        base_datasets_text = ""
        if self.base_datasets_info:
            base_datasets_text = "\n数据集→设备(推断来源):\n"
            for dataset_name, dataset_info in self.base_datasets_info.items():
                if isinstance(dataset_info, dict) and 'main_devices' in dataset_info:
                    devices = dataset_info['main_devices']
                    base_datasets_text += f"- {dataset_name}: {', '.join(devices)}\n"

        prompt = f"""{self.system_prompt}
{data_info_text}{base_datasets_text}
以下为 {len(predictions)} 个模型的预测(JSON)：

{formatted_preds}

选出最佳结果，严格按【输出】只回复 JSON。"""
        
        response_text = None
        try:
            # 智谱 GLM：关闭“思考”可避免 reasoning 占满 max_tokens；不支持的 API 会忽略或报错，则重试不带该参数
            kwargs = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                      "temperature": self.temperature, "max_tokens": self.max_tokens}
            try:
                completion = self.client.chat.completions.create(**kwargs, extra_body={"thinking": {"type": "disabled"}})
            except Exception:
                completion = self.client.chat.completions.create(**kwargs)
            if not completion.choices:
                print("✗ API 返回无 choices，使用降级选择")
                return self._fallback_selection(predictions)
            choice = completion.choices[0]
            msg = getattr(choice, "message", None) or choice
            response_text = (getattr(msg, "content", None) or "").strip() if msg else ""
            if not response_text:
                finish_reason = getattr(choice, "finish_reason", None) or getattr(msg, "finish_reason", None)
                print(f"✗ API 返回空内容 (finish_reason={finish_reason})，使用降级选择")
                return self._fallback_selection(predictions)
            # Extract JSON from response
            json_text = self._extract_json_from_text(response_text)
            decision_data = json.loads(json_text)
            
            # Validate required fields
            if "selected_model" not in decision_data:
                raise ValueError("响应中缺少 'selected_model' 字段")
            if "selected_class" not in decision_data:
                raise ValueError("响应中缺少 'selected_class' 字段")
            if "confidence" not in decision_data:
                raise ValueError("响应中缺少 'confidence' 字段")
            if "reasoning" not in decision_data:
                raise ValueError("响应中缺少 'reasoning' 字段")
            decision_data = self._post_check_structured_fields(decision_data, predictions)

            # Create AgentDecision
            decision = AgentDecision(
                selected_model=decision_data["selected_model"],
                selected_class=decision_data["selected_class"],
                confidence=float(decision_data["confidence"]),
                reasoning=decision_data["reasoning"],
                all_predictions=[pred.to_dict() for pred in predictions]
            )
            
            return decision
            
        except json.JSONDecodeError as e:
            print(f"✗ 无法解析 Gemini 响应为 JSON: {e}")
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
                print(f"   完整响应长度: {len(response_text)} 字符")
            print("   使用降级选择（选择最高置信度模型）")
            return self._fallback_selection(predictions)
        
        except KeyError as e:
            print(f"✗ Gemini 响应缺少必需字段: {e}")
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
            print("   使用降级选择（选择最高置信度模型）")
            return self._fallback_selection(predictions)
        
        except Exception as e:
            print(f"✗ 调用 Aliyun Bailian API 失败: {e}")
            import traceback
            traceback.print_exc()
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
            print("   使用降级选择（选择最高置信度模型）")
            return self._fallback_selection(predictions)
    
    def _fallback_selection(self, predictions: List[ModelOutput]) -> AgentDecision:
        """
        Fallback method to select best model if Gemini fails
        Selects the model with highest confidence, with additional validation
        
        Args:
            predictions: List of ModelOutput
            
        Returns:
            AgentDecision with highest confidence model
        """
        # Find highest confidence prediction
        best_pred = max(predictions, key=lambda p: p.top_confidence)
        
        # Count how many models agree with this prediction
        agreement_count = sum(
            1 for pred in predictions 
            if pred.top_class == best_pred.top_class and pred.top_confidence > 0.7
        )
        
        # Build reasoning
        if agreement_count >= 3:
            reasoning = f"降级选择：最高置信度模型（{best_pred.top_confidence:.2%}），{agreement_count}个模型一致预测为{best_pred.top_class}"
        else:
            reasoning = f"降级选择：最高置信度模型（{best_pred.top_confidence:.2%}）"
        
        return AgentDecision(
            selected_model=best_pred.model_name,
            selected_class=best_pred.top_class,
            confidence=best_pred.top_confidence,
            reasoning=reasoning,
            all_predictions=[pred.to_dict() for pred in predictions]
        )
    
    def batch_select(
        self,
        batch_predictions: List[List[ModelOutput]],
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> List[AgentDecision]:
        """
        Process multiple sets of predictions in batch (sequentially)
        
        Args:
            batch_predictions: List of prediction lists
            input_device_info: Optional list of device information for input data (applies to all predictions)
            
        Returns:
            List of AgentDecisions
        """
        decisions = []
        for predictions in batch_predictions:
            decision = self.select_best_model(
                predictions,
                input_device_info=input_device_info,
                input_data_info=input_data_info
            )
            decisions.append(decision)
        return decisions
    
    def select_best_model_batch(
        self,
        batch_data: List[Dict[str, Any]],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
        incremental_save_path: Optional[str] = None,
    ) -> List[AgentDecision]:
        """
        Use Gemini agent to select best model predictions for multiple images in a single API call
        This is more efficient than calling select_best_model for each image individually
        
        Args:
            batch_data: List of dicts with keys:
                - "image_name": str
                - "image_file": str 
                - "predictions": List[ModelOutput]
            use_json_format: Whether to format predictions as JSON
            input_device_info: Optional list of device information for input data (e.g., ["GE", "Siemens"])
                             If None, device information is unknown. This applies to all images in the batch.
            input_data_info: Optional full data context from config.data. null/None fields are treated as unknown.
            incremental_save_path: If set, after each batch write current results to this JSON file (same format as final results).
            
        Returns:
            List of AgentDecision objects, one for each image
        """
        if not batch_data:
            raise ValueError("No batch data provided")
        
        max_batch_size = self.max_batch_size
        
        def _save_incremental(decisions: List[AgentDecision], data: List[Dict[str, Any]], path: str) -> None:
            results = []
            for item, d in zip(data, decisions):
                preds = d.all_predictions or []
                results.append({
                    "image_file": item.get("image_file", ""),
                    "image_name": item.get("image_name", ""),
                    "selected_model": d.selected_model,
                    "predicted_class": d.selected_class,
                    "confidence": float(d.confidence),
                    "reasoning": d.reasoning,
                    "all_predictions": [
                        {
                            "model": p.get("model_name", ""),
                            "top_class": p.get("top_class", ""),
                            "top_confidence": float(p.get("top_confidence", 0)),
                            "predictions": {k: float(v) for k, v in (p.get("predictions") or {}).items()},
                        }
                        for p in preds
                    ],
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        if len(batch_data) > max_batch_size:
            print(f"   批量大小 ({len(batch_data)}) 超过限制 ({max_batch_size})，分批处理...")
            all_decisions = []
            for i in range(0, len(batch_data), max_batch_size):
                chunk = batch_data[i:i + max_batch_size]
                print(f"   处理批次 {i//max_batch_size + 1}/{(len(batch_data) + max_batch_size - 1)//max_batch_size} ({len(chunk)} 个图像)...")
                chunk_decisions = self._process_single_batch(
                    chunk,
                    use_json_format,
                    input_device_info,
                    input_data_info
                )
                all_decisions.extend(chunk_decisions)
                if incremental_save_path:
                    _save_incremental(all_decisions, batch_data[: len(all_decisions)], incremental_save_path)
            return all_decisions
        else:
            decisions = self._process_single_batch(
                batch_data,
                use_json_format,
                input_device_info,
                input_data_info
            )
            if incremental_save_path:
                _save_incremental(decisions, batch_data, incremental_save_path)
            return decisions
    
    def _process_single_batch(
        self,
        batch_data: List[Dict[str, Any]],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> List[AgentDecision]:
        """
        Process a single batch of images (internal method)
        
        Args:
            batch_data: List of dicts with prediction data
            use_json_format: Whether to format predictions as JSON
            input_device_info: Optional list of device information for input data
            input_data_info: Optional full data context from config.data
            
        Returns:
            List of AgentDecision objects

        对每张图：若该图各模型 top_class 一致则本地选最高置信度；仅对存在分歧的图调用大模型（子批次一次请求）。
        """
        decisions: List[Optional[AgentDecision]] = [None] * len(batch_data)
        need_llm_indices: List[int] = []
        for i, item in enumerate(batch_data):
            preds = item["predictions"]
            if self._predictions_top_class_unanimous(preds):
                decisions[i] = self._decision_unanimous_max_confidence(preds)
            else:
                need_llm_indices.append(i)

        if not need_llm_indices:
            return decisions  # type: ignore[return-value]

        sub_batch = [batch_data[i] for i in need_llm_indices]
        sub_decisions = self._process_single_batch_llm(
            sub_batch,
            use_json_format,
            input_device_info,
            input_data_info
        )
        for j, orig_i in enumerate(need_llm_indices):
            decisions[orig_i] = sub_decisions[j]
        return decisions  # type: ignore[return-value]

    def _process_single_batch_llm(
        self,
        batch_data: List[Dict[str, Any]],
        use_json_format: bool = True,
        input_device_info: Optional[List[str]] = None,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> List[AgentDecision]:
        """仅含存在类别分歧的样本，调用大模型批量决策。"""
        batch_system_prompt = """你是甲状腺超声多模型整合专家：对 batch 中每张图单独从多模型输出中选最可信项。

【单图规则】与单图任务相同：主置信度(top_confidence_calibrated 优先)→设备(已知时)→差<0.05 比 on_training_dataset 的 acc/AUC/F1→entropy↓ margin↑→base_dataset_performance/dataset_size→投票与高置信>0.95；字段见各 predictions 的 metadata。

【批处理】每图独立决策；仅当某模型在整批持续极端不合理时，可整体降低其权重。

【输出】纯 JSON，无 Markdown/思考。结构：
{"decisions":[{"image_index":0,"image_name":"","selected_model":"","selected_class":"","confidence":0.0,"runner_up_model":"","runner_up_confidence":0.0,"delta_confidence":0.0,"triggered_rules":["R1"],"reasoning":""},...]}。
decisions 长度必须等于图像数，顺序与输入 image_index 一致。
【一致性】delta_confidence=confidence-runner_up_confidence；delta>=0.05 才能写“显著高于/远高于”，否则写“高于/接近”。"""
        
        # Format batch data
        if use_json_format:
            formatted_data = {
                "num_images": len(batch_data),
                "images": []
            }
            
            for idx, item in enumerate(batch_data):
                image_data = {
                    "image_index": idx,  # Add index to ensure 1-to-1 mapping
                    "image_name": item["image_name"],
                    "image_file": item["image_file"],
                    "num_models": len(item["predictions"]),
                    "predictions": self._build_compact_prediction_dicts(item["predictions"])
                }
                formatted_data["images"].append(image_data)
            
            formatted_str = json.dumps(formatted_data, ensure_ascii=False)  # 不缩进以减小 prompt 体积
        else:
            # Text format
            formatted_str = f"# Batch Predictions for {len(batch_data)} Images\n\n"
            for idx, item in enumerate(batch_data, 1):
                formatted_str += f"## Image {idx}: {item['image_name']}\n\n"
                formatted_str += self.format_predictions(item["predictions"])
                formatted_str += "\n" + "-" * 70 + "\n\n"
        
        data_info_text = self._format_input_data_info_text(
            input_device_info=input_device_info,
            input_data_info=input_data_info
        )

        n_img = len(batch_data)
        prompt = f"""{batch_system_prompt}
{data_info_text}
共 {n_img} 张图的多模型预测(JSON)；decisions 必须恰好 {n_img} 条且与 image_index 顺序一致：

{formatted_str}

按【输出】只回复 JSON。"""
        
        response_text = None
        try:
            # 若模型开启“思考”会先占 2k+ token；关闭思考并预留 8192 以便输出完整 JSON
            max_tokens_batch = max(self.max_tokens, 8192)
            kwargs = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                      "temperature": self.temperature, "max_tokens": max_tokens_batch}
            try:
                completion = self.client.chat.completions.create(**kwargs, extra_body={"thinking": {"type": "disabled"}})
            except Exception:
                completion = self.client.chat.completions.create(**kwargs)
            if not completion.choices:
                print(f"   ✗ API 返回无 choices，回退到降级选择")
                return [self._fallback_selection(item["predictions"]) for item in batch_data]
            choice = completion.choices[0]
            msg = getattr(choice, "message", None) or choice
            response_text = (getattr(msg, "content", None) or "").strip() if msg else ""
            # 空响应时直接降级，避免解析报错
            if not response_text:
                finish_reason = getattr(choice, "finish_reason", None) or getattr(msg, "finish_reason", None)
                usage = getattr(completion, "usage", None)
                print(f"   ✗ API 返回空内容 (finish_reason={finish_reason}, usage={usage})，回退到降级选择")
                return [self._fallback_selection(item["predictions"]) for item in batch_data]
            # Extract JSON from response
            json_text = self._extract_json_from_text(response_text)
            # Debug: Print response size info
            print(f"   Gemini 响应长度: {len(response_text)} 字符")
            
            response_data = json.loads(json_text)
            
            # Debug: Print decisions count in response
            if "decisions" in response_data:
                print(f"   Gemini 返回的决策数量: {len(response_data['decisions'])}")
            
            # Validate response structure
            if "decisions" not in response_data:
                raise ValueError("响应中缺少 'decisions' 数组")
            
            decisions_list = response_data.get("decisions", [])
            if not isinstance(decisions_list, list):
                raise ValueError("'decisions' 字段必须是数组")
            
            # Validate decisions count matches batch size
            expected_count = len(batch_data)
            actual_count = len(decisions_list)
            
            if actual_count != expected_count:
                print(f"⚠️  警告: Agent 返回的决策数量 ({actual_count}) 与输入图像数量 ({expected_count}) 不匹配")
                
                # If more decisions than expected, try to deduplicate by image_name
                if actual_count > expected_count:
                    print(f"   尝试根据 image_name 去重...")
                    seen_names = set()
                    deduped_list = []
                    for decision_data in decisions_list:
                        img_name = decision_data.get("image_name", "")
                        if img_name not in seen_names:
                            seen_names.add(img_name)
                            deduped_list.append(decision_data)
                    
                    print(f"   去重后决策数量: {len(deduped_list)}")
                    
                    # If still too many, take first N
                    if len(deduped_list) > expected_count:
                        print(f"   仍然过多，将只使用前 {expected_count} 个决策")
                        decisions_list = deduped_list[:expected_count]
                    else:
                        decisions_list = deduped_list
                # If fewer decisions than expected, we'll handle it below
            
            # Create AgentDecision objects
            decisions = []
            for i, item in enumerate(batch_data):
                if i < len(decisions_list):
                    decision_data = decisions_list[i]
                    
                    # Validate decision data
                    required_fields = ["selected_model", "selected_class", "confidence", "reasoning"]
                    for field in required_fields:
                        if field not in decision_data:
                            raise ValueError(f"决策 {i} 中缺少 '{field}' 字段")
                    decision_data = self._post_check_structured_fields(decision_data, item["predictions"])

                    decision = AgentDecision(
                        selected_model=decision_data["selected_model"],
                        selected_class=decision_data["selected_class"],
                        confidence=float(decision_data["confidence"]),
                        reasoning=decision_data["reasoning"],
                        all_predictions=[pred.to_dict() for pred in item["predictions"]]
                    )
                    decisions.append(decision)
                else:
                    # If response is incomplete, use fallback for remaining images
                    print(f"⚠️  图像 {i+1} ({item['image_name']}) 的决策缺失，使用降级选择")
                    decisions.append(self._fallback_selection(item["predictions"]))
            
            return decisions
            
        except json.JSONDecodeError as e:
            print(f"✗ 无法解析 Gemini 批量响应为 JSON: {e}")
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
                print(f"   完整响应长度: {len(response_text)} 字符")
            print("   回退到单张处理模式（使用降级选择）...")
            return [self._fallback_selection(item["predictions"]) for item in batch_data]
        
        except KeyError as e:
            print(f"✗ Gemini 批量响应缺少必需字段: {e}")
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
            print("   回退到单张处理模式（使用降级选择）...")
            return [self._fallback_selection(item["predictions"]) for item in batch_data]
        
        except Exception as e:
            print(f"✗ 批量调用 Aliyun Bailian API 失败: {e}")
            import traceback
            traceback.print_exc()
            if response_text:
                print(f"   响应内容 (前500字符): {response_text[:500]}")
            print("   回退到单张处理模式（使用降级选择）...")
            return [self._fallback_selection(item["predictions"]) for item in batch_data]


#
# ============================================================
# 原始 Prompt 注释归档（文件末尾，纯注释；不参与运行）
# ============================================================
#
# SINGLE_IMAGE_PROMPT_ORIGINAL：
# 你是一个专业的甲状腺超声多模型结果整合专家，需要在多个分类模型的预测中选出最可靠的一项。
#
# 【设备先验（简要）】
# - 不同品牌/型号超声设备在对比度、分辨率和纹理上存在系统差异，模型在“训练时见过的设备”上通常更可靠。
# - GE 系列 (如 Logiq E9, S7) 风格相近；Hitachi 系列 (如 ARIETTA 850, Aloka Arietta V70) 风格相近；
# - 其他如 RESONA 70B、Toshiba Nemio 系列、Esaote 便携机等与上述设备存在风格差异；Heterogeneous 表示多设备混合来源，需要更强泛化能力。
#
# 【不确定性与一致性字段（来自 predictions.metadata）】
# - 主置信度相关：classification_uncertainty.top_confidence_calibrated（如存在优先使用）、classification_uncertainty.top_confidence_raw 或 top_confidence。
# - 不确定性：classification_uncertainty.entropy（以 2 为底，越大越不确定）、classification_uncertainty.margin_top2（top1 与 top2 概率差值，越大越稳定）。
# - 一致性：consistency_metrics.num_models_same_class（同一 top_class 的模型数）、consistency_metrics.total_models（模型总数）、consistency_metrics.vote_entropy（投票熵，越大意见越分散）。
#
# 【决策优先级】
# 1. 主置信度（最高优先级）
#    - 若存在 top_confidence_calibrated，用其作为主置信度；否则使用 top_confidence / top_confidence_raw。
#    - 其他条件相近时，优先主置信度更高的模型。
# 2. 设备匹配（重要）
#    - 如输入设备已知，优先训练数据包含相同或同品牌设备的模型；如输入设备未知 (null)，则跳过该因素。
# 3. 验证集性能（重要）
#    - 当主置信度差异 < 0.05 时，对比 validation_metrics.on_training_dataset 的 accuracy / AUC / f1_score，优先性能更高的模型，尤其是在对应原始数据集测试集上的表现。
# 4. 分类不确定性（重要）
#    - 在主置信度和验证性能接近时，优先 entropy 更小、margin_top2 更大的模型。
# 5. 数据集规模与来源（次要）
#    - 若可根据设备信息推断输入可能来源的原始数据集 (TN3K / ThyroidXL / TN5K / CineClip)，优先在该数据集上 base_dataset_performance 更好的模型。
#    - 若无法推断来源且多个模型接近，则优先 dataset_size 更大的模型（泛化能力通常更强）。
# 6. 模型一致性
#    - 若有 ≥3 个模型对同一类别给出高置信度（例如 >0.85），且该类别的 num_models_same_class 较高、vote_entropy 较低，可增强该类别的可信度。
# 7. 置信度差异与多数投票
#    - 当最高主置信度与次高主置信度差异 < 0.05 时，结合各类别的投票数量和 num_models_same_class 进行多数投票判断。
# 8. 模型特性（兜底）
#    - 仅在上述指标都极为接近（主置信度差异 < 0.02）时，再考虑模型是否为特定任务/结构变体。
#
# 【重要原则】
# - 以“（校准后）主置信度 + 较低不确定性”为主导，结合设备匹配和验证性能综合决策。
# - 设备匹配与原始数据集性能用于解释“为什么在该设备/数据集上更可信”，优先级低于主置信度但高于纯模型名称或结构差异。
# - 更大且覆盖面更广的数据集一般意味着更好泛化能力，但优先级低于置信度与验证性能。
# - 当多个模型预测不同类别时，如存在置信度 >0.95 的预测，一般优先信任该预测。
#
# 【输出要求】
# 你必须用中文返回一个**纯 JSON 对象**（不要包含思考过程、说明文字或 Markdown 代码块），格式为：
# {
#   "selected_model": "最佳模型名称",
#   "selected_class": "类别名称",
#   "confidence": 0.0~1.0 的数值,
#   "reasoning": "3~4 句中文，按上述逻辑给出关键数值和理由"
# }
#
# 对 reasoning 的具体要求：
# - 使用客观、学术风格表述，不使用“位居榜首”“脱颖而出”等修辞。
# - 尽量用描述性称呼（如“置信度最高的模型”“在匹配设备数据上训练的模型”），避免频繁直呼具体模型名。
# - 明确说明：采用的主置信度及其数值、entropy 和 margin_top2 的数值、num_models_same_class 和 vote_entropy、验证集 accuracy/AUC/f1_score 及与其他模型的主要差值、dataset_size 与包含的原始数据集，以及这些因素如何共同支持你的选择。
# - 简要对比最主要的 1–2 个竞争模型（例如置信度差 0.03、在某原始数据集的 AUC 高约 0.02），避免“显著更好”等模糊描述。
# 只输出上述 JSON，对应字段必须齐全。
#
# BATCH_PROMPT_ORIGINAL：
# 你是一个专业的甲状腺超声多模型结果整合专家，需要在多张图像上，根据多个分类模型的预测结果，为每张图像选出最可靠的一项。
#
# 【设备先验与基本概念】
# - 设备差异与训练时见过的设备会显著影响模型可靠性；GE 系列与 Hitachi 系列内部风格相近，其他品牌及便携设备与其存在不同风格或为多设备混合 (Heterogeneous)。
# - 同一张图像上，不同模型的预测可通过置信度、不确定性和一致性进行综合比较。
#
# 【可用字段】
# - 置信度与不确定性：top_confidence_calibrated、top_confidence_raw/top_confidence、entropy、margin_top2。
# - 一致性：num_models_same_class、total_models、vote_entropy。
# - 性能与数据集：validation_metrics.on_training_dataset（accuracy / AUC / f1_score）、base_dataset_performance、dataset_info.base_datasets、dataset_info.dataset_size。
#
# 【单张图像决策优先级】（与单图像版本一致）
# 1. 主置信度（最高优先级）
#    - 若有 top_confidence_calibrated，用其作为主置信度；否则用 top_confidence / top_confidence_raw。
# 2. 设备匹配
# 3. 验证集性能（当主置信度差异 < 0.05 时比对 acc/AUC/F1）
# 4. 不确定性（entropy 更小、margin_top2 更大）
# 5. 数据集规模与来源（能推断则看 base_dataset_performance，否则 dataset_size 更大优先）
# 6. 模型一致性（num_models_same_class 与 vote_entropy）
# 7. 模型特性（兜底）
#
# 【跨图像注意点】
# - 每张图像主要依据自身指标；只有当某模型在整批样本上持续极端不合理时，才可整体降低权重。
# - 仍需为每张图像给出清晰、数据驱动的理由。
#
# 【输出要求】
# 你必须返回一个**纯 JSON 对象**（不要包含思考过程或 Markdown），结构：
# {
#   "decisions": [
#     {
#       "image_index": 0,
#       "image_name": "...",
#       "selected_model": "...",
#       "selected_class": "...",
#       "confidence": 0.0~1.0,
#       "reasoning": "3~4 句中文说明"
#     }
#   ]
# }
#
# 约束：
# - decisions 数组长度必须恰好等于输入图像数量，顺序与 image_index 一致。
# - 每个元素必须包含 image_index、image_name、selected_model、selected_class、confidence、reasoning。
# - reasoning：结合主置信度对比、entropy/margin_top2、设备匹配、验证集差异、dataset_size/base_datasets、一致性指标等，且只简要对比 1–2 个竞争模型。
#
# SINGLE_IMAGE_USER_PROMPT_TAIL_ORIGINAL：
# 以下是来自 {len(predictions)} 个不同模型的预测结果：
#
# {formatted_preds}
#
# 基于上述预测结果，哪个模型提供了最佳的分类结果？
#
# 直接返回 JSON（不带 Markdown/思考/代码块），reasoning 字段中文，首字符必须为 {，末字符必须为 }。
#
# BATCH_USER_PROMPT_TAIL_ORIGINAL：
# 以下是多个模型对 {len(batch_data)} 张图像的预测结果：
#
# {formatted_str}
#
# 基于上述预测结果，为每张图像选择最佳模型。
#
# 必须返回纯 JSON，不带 Markdown/思考/代码块；decisions 长度必须恰好为图像数且顺序与 image_index 一致；首字符 {、末字符 }。

