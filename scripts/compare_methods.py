"""
分类性能对比脚本
对比 Agent 方法和固定方法（选择最高置信度）的分类性能

固定方法: 对每个图像，选取所有模型中置信度最高的分类结果
Agent方法: 使用 Gemini Agent 综合决策的分类结果
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class ClassificationMetrics:
    """分类性能指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    confusion_matrix: Dict[str, Dict[str, int]] = None
    correct_count: int = 0
    total_count: int = 0
    
    def __str__(self):
        result = (
            f"准确率 (Accuracy): {self.accuracy:.4f} ({self.correct_count}/{self.total_count})\n"
            f"精确率 (Precision): {self.precision:.4f}\n"
            f"召回率 (Recall): {self.recall:.4f}\n"
            f"F1分数: {self.f1_score:.4f}"
        )
        if self.auroc is not None:
            result += f"\nAUROC: {self.auroc:.4f}"
        if self.auprc is not None:
            result += f"\nAUPRC: {self.auprc:.4f}"
        if self.sensitivity is not None:
            result += f"\n敏感性 (Sensitivity): {self.sensitivity:.4f}"
        if self.specificity is not None:
            result += f"\n特异性 (Specificity): {self.specificity:.4f}"
        return result


def load_labels(label_file: str) -> Dict[str, str]:
    """
    加载分类标签文件
    
    支持JSON格式，格式如下:
    [
      {
        "filename": "image_name.jpg",
        "malignancy": 0,  # 0=良性, 1=恶性
        "tirads": 3  # 可选字段
      },
      ...
    ]
    
    Args:
        label_file: 标签文件路径（必须是JSON格式）
        
    Returns:
        图像名称到标签的映射字典（标签已转换为模型输出格式："良性"或"恶性"）
    """
    label_path = Path(label_file)
    
    if not label_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_file}")
    
    if label_path.suffix.lower() != '.json':
        raise ValueError(f"标签文件必须是JSON格式，当前文件: {label_file}")
    
    # 加载JSON文件
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = {}
    
    # 检查是数组格式还是对象格式
    if isinstance(data, list):
        # 新格式：JSON数组，每个元素包含 filename 和 malignancy
        for item in data:
            if not isinstance(item, dict):
                print(f"⚠️  跳过无效的数组元素: {item}")
                continue
            
            if "filename" not in item:
                print(f"⚠️  跳过缺少 'filename' 字段的元素: {item}")
                continue
            
            if "malignancy" not in item:
                print(f"⚠️  跳过缺少 'malignancy' 字段的元素: {item}")
                continue
            
            filename = item["filename"]
            malignancy = item["malignancy"]
            
            # 将数字转换为字符串以便统一处理
            if isinstance(malignancy, int):
                malignancy_str = str(malignancy)
            elif isinstance(malignancy, str):
                malignancy_str = malignancy
            else:
                print(f"⚠️  跳过无效的 malignancy 值 '{malignancy}' (图像: {filename})")
                continue
            
            labels[filename] = malignancy_str
            
            # tirads 字段可选，这里不处理（只用于良恶性分类对比）
    
    elif isinstance(data, dict):
        # 旧格式兼容：对象格式 {"image_name": "label", ...}
        labels = data
    else:
        raise ValueError(f"不支持的JSON格式，期望数组或对象，得到: {type(data)}")
    
    # 标签映射：将数字标签转换为中文标签（与模型输出格式一致）
    label_mapping = {
        "0": "良性",
        "1": "恶性",
        0: "良性",  # 支持整数类型
        1: "恶性",
        "良性": "良性",  # 保持中文标签不变
        "恶性": "恶性"
    }
    
    # 转换标签
    converted_labels = {}
    unmapped_labels = set()
    
    for image_name, label in labels.items():
        # 处理整数类型 - 直接在这里转换
        if isinstance(label, int):
            if label in label_mapping:
                converted_labels[image_name] = label_mapping[label]
            else:
                unmapped_labels.add(label)
                converted_labels[image_name] = str(label)
        elif isinstance(label, str):
            if label in label_mapping:
                converted_labels[image_name] = label_mapping[label]
            else:
                unmapped_labels.add(label)
                converted_labels[image_name] = label
        else:
            unmapped_labels.add(label)
            converted_labels[image_name] = str(label)
    
    if unmapped_labels:
        print(f"⚠️  警告: 发现未映射的标签格式: {sorted(unmapped_labels)}")
    
    print(f"✓ 成功加载 {len(converted_labels)} 个标签")
    
    # 显示标签分布
    label_counts = {}
    for label in converted_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"   标签分布: {label_counts}")
    
    return converted_labels


def load_agent_results(result_file: str) -> Dict[str, Dict]:
    """
    加载 Agent 的输出结果
    
    Args:
        result_file: Agent输出的JSON结果文件路径
        
    Returns:
        图像名称到结果的映射字典
    """
    result_path = Path(result_file)
    
    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_file}")
    
    with open(result_path, 'r', encoding='utf-8') as f:
        results_list = json.load(f)
    
    # 转换为字典格式，使用image_name作为key
    results = {}
    for item in results_list:
        image_name = item['image_name']
        results[image_name] = item
    
    print(f"✓ 成功加载 {len(results)} 个Agent决策结果")
    return results


def get_max_confidence_prediction(all_predictions: List[Dict]) -> Tuple[str, float, str, Dict[str, float]]:
    """
    从所有模型预测中选择置信度最高的结果（固定方法）
    
    Args:
        all_predictions: 所有模型的预测结果列表
        
    Returns:
        (预测类别, 置信度, 模型名称, 概率字典)
    """
    max_confidence = -1
    best_class = None
    best_model = None
    best_predictions = None
    
    for pred in all_predictions:
        if pred['top_confidence'] > max_confidence:
            max_confidence = pred['top_confidence']
            best_class = pred['top_class']
            best_model = pred['model']
            best_predictions = pred.get('predictions', {})
    
    return best_class, max_confidence, best_model, best_predictions or {}


def extract_probabilities_from_agent_results(agent_results: Dict[str, Dict], 
                                             class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    从 Agent 结果中提取每个图像的预测概率
    
    Args:
        agent_results: Agent结果字典
        class_names: 类别名称列表
        
    Returns:
        图像名称到概率字典的映射（概率字典的键为类别名称）
    """
    probabilities = {}
    
    for image_name, result in agent_results.items():
        # Agent方法：使用选中模型的概率分布
        selected_model = result.get('selected_model', '')
        all_predictions = result.get('all_predictions', [])
        
        # 找到选中模型的预测结果
        selected_pred = None
        for pred in all_predictions:
            if pred.get('model') == selected_model:
                selected_pred = pred
                break
        
        if selected_pred and 'predictions' in selected_pred:
            probabilities[image_name] = selected_pred['predictions']
        else:
            # 如果找不到，使用置信度构建一个简单的概率分布
            # 假设只有两个类别
            pred_class = result.get('predicted_class', '')
            confidence = result.get('confidence', 0.5)
            if len(class_names) == 2:
                prob_dict = {}
                for cls in class_names:
                    if cls == pred_class:
                        prob_dict[cls] = confidence
                    else:
                        prob_dict[cls] = 1.0 - confidence
                probabilities[image_name] = prob_dict
            else:
                # 多分类情况，使用均匀分布作为后备
                probabilities[image_name] = {cls: 1.0 / len(class_names) for cls in class_names}
    
    return probabilities


def extract_probabilities_from_fixed_method(agent_results: Dict[str, Dict],
                                           class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    从固定方法（最高置信度）中提取每个图像的预测概率
    
    Args:
        agent_results: Agent结果字典（包含all_predictions）
        class_names: 类别名称列表
        
    Returns:
        图像名称到概率字典的映射
    """
    probabilities = {}
    
    for image_name, result in agent_results.items():
        all_predictions = result.get('all_predictions', [])
        _, _, _, best_predictions = get_max_confidence_prediction(all_predictions)
        
        if best_predictions:
            probabilities[image_name] = best_predictions
        else:
            # 后备方案
            if len(class_names) == 2:
                probabilities[image_name] = {cls: 0.5 for cls in class_names}
            else:
                probabilities[image_name] = {cls: 1.0 / len(class_names) for cls in class_names}
    
    return probabilities


def calculate_metrics(predictions: Dict[str, str], 
                     labels: Dict[str, str],
                     class_names: List[str],
                     probabilities: Optional[Dict[str, Dict[str, float]]] = None) -> ClassificationMetrics:
    """
    计算分类性能指标
    
    Args:
        predictions: 图像名称到预测类别的映射
        labels: 图像名称到真实标签的映射
        class_names: 所有类别名称列表
        probabilities: 可选的，图像名称到概率字典的映射（用于计算AUROC和AUPRC）
        
    Returns:
        ClassificationMetrics对象
    """
    # 过滤出有标签的图像
    common_images = set(predictions.keys()) & set(labels.keys())
    
    if len(common_images) == 0:
        raise ValueError("预测结果和标签文件没有交集，请检查图像名称是否匹配")
    
    # 初始化混淆矩阵
    confusion_matrix = {true_class: {pred_class: 0 for pred_class in class_names} 
                       for true_class in class_names}
    
    # 统计各类别的TP, FP, FN
    tp = {cls: 0 for cls in class_names}
    fp = {cls: 0 for cls in class_names}
    fn = {cls: 0 for cls in class_names}
    tn = {cls: 0 for cls in class_names}  # True Negative，用于计算Specificity
    
    correct_count = 0
    total_count = len(common_images)
    
    for image_name in common_images:
        pred = predictions[image_name]
        true_label = labels[image_name]
        
        # 更新混淆矩阵
        confusion_matrix[true_label][pred] += 1
        
        # 统计正确数量
        if pred == true_label:
            correct_count += 1
            tp[pred] += 1
            # 对于其他类别，这是True Negative
            for cls in class_names:
                if cls != pred:
                    tn[cls] += 1
        else:
            fp[pred] += 1
            fn[true_label] += 1
            # 对于其他既不是pred也不是true_label的类别，这是TN
            for cls in class_names:
                if cls != pred and cls != true_label:
                    tn[cls] += 1
    
    # 计算整体指标
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 对于二分类，按照 utils/metrics.py 的方式计算正类（恶性）的指标
    # 对于多分类，使用宏平均
    if len(class_names) == 2:
        # 二分类：使用第二个类别作为正类（通常是"恶性"）
        positive_class = class_names[1]
        negative_class = class_names[0]
        
        # 计算正类（恶性）的 TP, FP, FN, TN
        # TP: 真实是正类，预测也是正类
        tp_positive = tp[positive_class]
        # FP: 真实是负类，但预测为正类
        fp_positive = fp[positive_class]
        # FN: 真实是正类，但预测为负类
        fn_positive = fn[positive_class]
        # TN: 真实是负类，预测也是负类（即负类的TP）
        tn_positive = tp[negative_class]
        
        # 计算正类（恶性）的精确率、召回率、F1分数
        # 与 utils/metrics.py 中的计算方式一致
        avg_precision = tp_positive / (tp_positive + fp_positive) if (tp_positive + fp_positive) > 0 else 0
        avg_recall = tp_positive / (tp_positive + fn_positive) if (tp_positive + fn_positive) > 0 else 0
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Sensitivity = Recall（正类的召回率）
        avg_sensitivity = avg_recall
        
        # Specificity = TN / (TN + FP) for positive class
        avg_specificity = tn_positive / (tn_positive + fp_positive) if (tn_positive + fp_positive) > 0 else 0
    else:
        # 多分类：使用宏平均
        precisions = []
        recalls = []
        f1_scores = []
        sensitivities = []
        specificities = []
        
        for cls in class_names:
            tp_cls = tp[cls]
            fp_cls = fp[cls]
            fn_cls = fn[cls]
            tn_cls = tn[cls]
            
            precision = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0
            recall = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Sensitivity = Recall (对于该类)
            sensitivity = recall
            # Specificity = TN / (TN + FP) (对于该类)
            specificity = tn_cls / (tn_cls + fp_cls) if (tn_cls + fp_cls) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
    
    # 计算AUROC和AUPRC（如果提供了概率值）
    auroc = None
    auprc = None
    
    if probabilities is not None:
        # 对于二分类，使用第一个类别作为正类
        # 对于多分类，计算宏平均的AUROC和AUPRC
        if len(class_names) == 2:
            # 二分类：使用第二个类别（通常是"恶性"）作为正类
            positive_class = class_names[1] if len(class_names) >= 2 else class_names[0]
            
            y_true_binary = []
            y_proba_positive = []
            
            for image_name in sorted(common_images):
                true_label = labels[image_name]
                # 转换为二分类标签：1表示正类，0表示负类
                y_true_binary.append(1 if true_label == positive_class else 0)
                
                # 获取正类的概率
                prob_dict = probabilities.get(image_name, {})
                prob_positive = prob_dict.get(positive_class, 0.5)
                y_proba_positive.append(prob_positive)
            
            try:
                if len(set(y_true_binary)) > 1:  # 确保有正负两类
                    auroc = roc_auc_score(y_true_binary, y_proba_positive)
                    auprc = average_precision_score(y_true_binary, y_proba_positive)
            except Exception as e:
                print(f"⚠️  计算AUROC/AUPRC时出错: {e}")
                auroc = None
                auprc = None
        else:
            # 多分类：计算宏平均的AUROC和AUPRC
            auroc_scores = []
            auprc_scores = []
            
            for positive_class in class_names:
                y_true_binary = []
                y_proba_positive = []
                
                for image_name in sorted(common_images):
                    true_label = labels[image_name]
                    y_true_binary.append(1 if true_label == positive_class else 0)
                    
                    prob_dict = probabilities.get(image_name, {})
                    prob_positive = prob_dict.get(positive_class, 1.0 / len(class_names))
                    y_proba_positive.append(prob_positive)
                
                try:
                    if len(set(y_true_binary)) > 1:
                        auroc_scores.append(roc_auc_score(y_true_binary, y_proba_positive))
                        auprc_scores.append(average_precision_score(y_true_binary, y_proba_positive))
                except Exception:
                    pass
            
            if auroc_scores:
                auroc = np.mean(auroc_scores)
            if auprc_scores:
                auprc = np.mean(auprc_scores)
    
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=avg_precision,
        recall=avg_recall,
        f1_score=avg_f1,
        auroc=auroc,
        auprc=auprc,
        sensitivity=avg_sensitivity,
        specificity=avg_specificity,
        confusion_matrix=confusion_matrix,
        correct_count=correct_count,
        total_count=total_count
    )


def print_confusion_matrix(confusion_matrix: Dict[str, Dict[str, int]], 
                          class_names: List[str], 
                          title: str = "混淆矩阵"):
    """打印混淆矩阵"""
    print(f"\n{title}")
    print("=" * 60)
    
    # 打印表头
    header = "真实\\预测".ljust(12)
    for cls in class_names:
        header += f"{cls:>10}"
    print(header)
    print("-" * 60)
    
    # 打印每一行
    for true_cls in class_names:
        row = f"{true_cls}".ljust(12)
        for pred_cls in class_names:
            count = confusion_matrix[true_cls][pred_cls]
            row += f"{count:>10}"
        print(row)
    print("=" * 60)


def analyze_disagreements(agent_results: Dict[str, Dict],
                         labels: Dict[str, str]) -> None:
    """
    分析 Agent 方法和固定方法的分歧情况
    
    Args:
        agent_results: Agent结果字典
        labels: 标签字典
    """
    print("\n" + "=" * 70)
    print("分歧分析 (Agent vs 固定方法)")
    print("=" * 70)
    
    disagreements = []
    agent_correct_fixed_wrong = []
    fixed_correct_agent_wrong = []
    both_wrong_but_different = []
    
    for image_name, result in agent_results.items():
        if image_name not in labels:
            continue
        
        true_label = labels[image_name]
        agent_pred = result['predicted_class']
        
        # 获取固定方法的预测
        fixed_pred, fixed_conf, fixed_model, _ = get_max_confidence_prediction(
            result['all_predictions']
        )
        
        # 如果两种方法预测不同
        if agent_pred != fixed_pred:
            disagreement_info = {
                'image_name': image_name,
                'true_label': true_label,
                'agent_pred': agent_pred,
                'agent_conf': result['confidence'],
                'agent_model': result['selected_model'],
                'fixed_pred': fixed_pred,
                'fixed_conf': fixed_conf,
                'fixed_model': fixed_model,
                'agent_correct': agent_pred == true_label,
                'fixed_correct': fixed_pred == true_label
            }
            
            disagreements.append(disagreement_info)
            
            if agent_pred == true_label and fixed_pred != true_label:
                agent_correct_fixed_wrong.append(disagreement_info)
            elif fixed_pred == true_label and agent_pred != true_label:
                fixed_correct_agent_wrong.append(disagreement_info)
            elif agent_pred != true_label and fixed_pred != true_label:
                both_wrong_but_different.append(disagreement_info)
    
    print(f"\n总分歧数量: {len(disagreements)}")
    print(f"  - Agent正确，固定方法错误: {len(agent_correct_fixed_wrong)}")
    print(f"  - 固定方法正确，Agent错误: {len(fixed_correct_agent_wrong)}")
    print(f"  - 两者都错但预测不同: {len(both_wrong_but_different)}")
    
    # 显示 Agent 正确而固定方法错误的案例
    if agent_correct_fixed_wrong:
        print(f"\n【Agent优于固定方法的案例】({len(agent_correct_fixed_wrong)}个)")
        print("-" * 70)
        for i, case in enumerate(agent_correct_fixed_wrong[:5], 1):  # 只显示前5个
            print(f"\n{i}. {case['image_name']}")
            print(f"   真实标签: {case['true_label']}")
            print(f"   Agent预测: {case['agent_pred']} (置信度: {case['agent_conf']:.4f}, 模型: {case['agent_model']})")
            print(f"   固定方法: {case['fixed_pred']} (置信度: {case['fixed_conf']:.4f}, 模型: {case['fixed_model']})")
        
        if len(agent_correct_fixed_wrong) > 5:
            print(f"\n   ... 还有 {len(agent_correct_fixed_wrong) - 5} 个类似案例")
    
    # 显示固定方法正确而 Agent 错误的案例
    if fixed_correct_agent_wrong:
        print(f"\n【固定方法优于Agent的案例】({len(fixed_correct_agent_wrong)}个)")
        print("-" * 70)
        for i, case in enumerate(fixed_correct_agent_wrong[:5], 1):
            print(f"\n{i}. {case['image_name']}")
            print(f"   真实标签: {case['true_label']}")
            print(f"   Agent预测: {case['agent_pred']} (置信度: {case['agent_conf']:.4f}, 模型: {case['agent_model']})")
            print(f"   固定方法: {case['fixed_pred']} (置信度: {case['fixed_conf']:.4f}, 模型: {case['fixed_model']})")
        
        if len(fixed_correct_agent_wrong) > 5:
            print(f"\n   ... 还有 {len(fixed_correct_agent_wrong) - 5} 个类似案例")


def save_comparison_report(agent_metrics: ClassificationMetrics,
                          fixed_metrics: ClassificationMetrics,
                          agent_predictions: Dict[str, str],
                          fixed_predictions: Dict[str, str],
                          labels: Dict[str, str],
                          output_file: str) -> None:
    """
    保存详细的对比报告
    
    Args:
        agent_metrics: Agent方法的性能指标
        fixed_metrics: 固定方法的性能指标
        agent_predictions: Agent的预测结果
        fixed_predictions: 固定方法的预测结果
        labels: 真实标签
        output_file: 输出文件路径
    """
    report = {
        "summary": {
            "agent_method": {
                "accuracy": float(agent_metrics.accuracy),
                "precision": float(agent_metrics.precision),
                "recall": float(agent_metrics.recall),
                "f1_score": float(agent_metrics.f1_score),
                "correct_count": agent_metrics.correct_count,
                "total_count": agent_metrics.total_count
            },
            "fixed_method": {
                "accuracy": float(fixed_metrics.accuracy),
                "precision": float(fixed_metrics.precision),
                "recall": float(fixed_metrics.recall),
                "f1_score": float(fixed_metrics.f1_score),
                "correct_count": fixed_metrics.correct_count,
                "total_count": fixed_metrics.total_count
            },
            "improvement": {
                "accuracy_diff": float(agent_metrics.accuracy - fixed_metrics.accuracy),
                "precision_diff": float(agent_metrics.precision - fixed_metrics.precision),
                "recall_diff": float(agent_metrics.recall - fixed_metrics.recall),
                "f1_diff": float(agent_metrics.f1_score - fixed_metrics.f1_score)
            }
        },
        "confusion_matrices": {
            "agent_method": agent_metrics.confusion_matrix,
            "fixed_method": fixed_metrics.confusion_matrix
        },
        "detailed_results": []
    }
    
    # 添加新指标到报告
    if agent_metrics.auroc is not None:
        report["summary"]["agent_method"]["auroc"] = float(agent_metrics.auroc)
    if agent_metrics.auprc is not None:
        report["summary"]["agent_method"]["auprc"] = float(agent_metrics.auprc)
    if agent_metrics.sensitivity is not None:
        report["summary"]["agent_method"]["sensitivity"] = float(agent_metrics.sensitivity)
    if agent_metrics.specificity is not None:
        report["summary"]["agent_method"]["specificity"] = float(agent_metrics.specificity)
    
    if fixed_metrics.auroc is not None:
        report["summary"]["fixed_method"]["auroc"] = float(fixed_metrics.auroc)
    if fixed_metrics.auprc is not None:
        report["summary"]["fixed_method"]["auprc"] = float(fixed_metrics.auprc)
    if fixed_metrics.sensitivity is not None:
        report["summary"]["fixed_method"]["sensitivity"] = float(fixed_metrics.sensitivity)
    if fixed_metrics.specificity is not None:
        report["summary"]["fixed_method"]["specificity"] = float(fixed_metrics.specificity)
    
    # 添加新指标的提升
    if agent_metrics.auroc is not None and fixed_metrics.auroc is not None:
        report["summary"]["improvement"]["auroc_diff"] = float(agent_metrics.auroc - fixed_metrics.auroc)
    if agent_metrics.auprc is not None and fixed_metrics.auprc is not None:
        report["summary"]["improvement"]["auprc_diff"] = float(agent_metrics.auprc - fixed_metrics.auprc)
    if agent_metrics.sensitivity is not None and fixed_metrics.sensitivity is not None:
        report["summary"]["improvement"]["sensitivity_diff"] = float(agent_metrics.sensitivity - fixed_metrics.sensitivity)
    if agent_metrics.specificity is not None and fixed_metrics.specificity is not None:
        report["summary"]["improvement"]["specificity_diff"] = float(agent_metrics.specificity - fixed_metrics.specificity)
    
    # 添加每个图像的详细结果
    common_images = set(agent_predictions.keys()) & set(labels.keys())
    for image_name in sorted(common_images):
        report["detailed_results"].append({
            "image_name": image_name,
            "true_label": labels[image_name],
            "agent_prediction": agent_predictions[image_name],
            "fixed_prediction": fixed_predictions[image_name],
            "agent_correct": agent_predictions[image_name] == labels[image_name],
            "fixed_correct": fixed_predictions[image_name] == labels[image_name]
        })
    
    # 保存为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 详细对比报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='对比 Agent 方法和固定方法（最高置信度）的分类性能'
    )
    parser.add_argument(
        'result_file',
        type=str,
        help='Agent输出的结果文件路径 (JSON格式)'
    )
    parser.add_argument(
        'label_file',
        type=str,
        help='分类标签文件路径 (JSON或CSV格式)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='对比报告输出路径 (默认: comparison_report_<timestamp>.json)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("分类性能对比工具")
    print("=" * 70)
    print()
    
    # 1. 加载数据
    print(">>> 步骤 1/4: 加载数据")
    print(f"    结果文件: {args.result_file}")
    print(f"    标签文件: {args.label_file}")
    
    labels = load_labels(args.label_file)
    agent_results = load_agent_results(args.result_file)
    
    # 获取所有类别名称
    class_names = sorted(set(labels.values()))
    print(f"    检测到类别: {class_names}")
    print()
    
    # 2. 提取预测结果
    print(">>> 步骤 2/4: 提取预测结果")
    
    # Agent的预测结果
    agent_predictions = {}
    for image_name, result in agent_results.items():
        agent_predictions[image_name] = result['predicted_class']
    
    # 固定方法的预测结果（选择最高置信度）
    fixed_predictions = {}
    fixed_details = {}  # 保存详细信息用于分析
    
    for image_name, result in agent_results.items():
        pred_class, confidence, model_name, _ = get_max_confidence_prediction(
            result['all_predictions']
        )
        fixed_predictions[image_name] = pred_class
        fixed_details[image_name] = {
            'class': pred_class,
            'confidence': confidence,
            'model': model_name
        }
    
    print(f"✓ Agent方法预测: {len(agent_predictions)} 个图像")
    print(f"✓ 固定方法预测: {len(fixed_predictions)} 个图像")
    
    # 更新类别名称，包含所有预测结果中的类别
    all_classes = set(labels.values()) | set(agent_predictions.values()) | set(fixed_predictions.values())
    class_names = sorted(all_classes)
    print(f"    最终类别列表（含预测）: {class_names}")
    print()
    
    # 提取概率值（用于计算AUROC和AUPRC）
    print(">>> 步骤 2.5/4: 提取预测概率")
    agent_probabilities = extract_probabilities_from_agent_results(agent_results, class_names)
    fixed_probabilities = extract_probabilities_from_fixed_method(agent_results, class_names)
    print(f"✓ Agent方法概率: {len(agent_probabilities)} 个图像")
    print(f"✓ 固定方法概率: {len(fixed_probabilities)} 个图像")
    print()
    
    # 3. 计算性能指标
    print(">>> 步骤 3/4: 计算性能指标")
    
    agent_metrics = calculate_metrics(agent_predictions, labels, class_names, agent_probabilities)
    fixed_metrics = calculate_metrics(fixed_predictions, labels, class_names, fixed_probabilities)
    
    print("✓ 性能指标计算完成")
    print()
    
    # 4. 显示对比结果
    print("=" * 70)
    print("性能对比结果")
    print("=" * 70)
    
    print(f"\n【Agent 方法】")
    print("-" * 70)
    print(agent_metrics)
    print_confusion_matrix(agent_metrics.confusion_matrix, class_names, "Agent方法 - 混淆矩阵")
    
    print(f"\n【固定方法 (最高置信度)】")
    print("-" * 70)
    print(fixed_metrics)
    print_confusion_matrix(fixed_metrics.confusion_matrix, class_names, "固定方法 - 混淆矩阵")
    
    # 性能提升对比
    print(f"\n【性能提升对比】")
    print("=" * 70)
    print(f"准确率提升: {(agent_metrics.accuracy - fixed_metrics.accuracy):+.4f} "
          f"({agent_metrics.accuracy:.4f} vs {fixed_metrics.accuracy:.4f})")
    print(f"精确率提升: {(agent_metrics.precision - fixed_metrics.precision):+.4f} "
          f"({agent_metrics.precision:.4f} vs {fixed_metrics.precision:.4f})")
    print(f"召回率提升: {(agent_metrics.recall - fixed_metrics.recall):+.4f} "
          f"({agent_metrics.recall:.4f} vs {fixed_metrics.recall:.4f})")
    print(f"F1分数提升: {(agent_metrics.f1_score - fixed_metrics.f1_score):+.4f} "
          f"({agent_metrics.f1_score:.4f} vs {fixed_metrics.f1_score:.4f})")
    if agent_metrics.auroc is not None and fixed_metrics.auroc is not None:
        print(f"AUROC提升: {(agent_metrics.auroc - fixed_metrics.auroc):+.4f} "
              f"({agent_metrics.auroc:.4f} vs {fixed_metrics.auroc:.4f})")
    if agent_metrics.auprc is not None and fixed_metrics.auprc is not None:
        print(f"AUPRC提升: {(agent_metrics.auprc - fixed_metrics.auprc):+.4f} "
              f"({agent_metrics.auprc:.4f} vs {fixed_metrics.auprc:.4f})")
    if agent_metrics.sensitivity is not None and fixed_metrics.sensitivity is not None:
        print(f"敏感性提升: {(agent_metrics.sensitivity - fixed_metrics.sensitivity):+.4f} "
              f"({agent_metrics.sensitivity:.4f} vs {fixed_metrics.sensitivity:.4f})")
    if agent_metrics.specificity is not None and fixed_metrics.specificity is not None:
        print(f"特异性提升: {(agent_metrics.specificity - fixed_metrics.specificity):+.4f} "
              f"({agent_metrics.specificity:.4f} vs {fixed_metrics.specificity:.4f})")
    
    # 判断哪种方法更好
    print(f"\n【结论】")
    print("=" * 70)
    if agent_metrics.accuracy > fixed_metrics.accuracy:
        improvement = (agent_metrics.accuracy - fixed_metrics.accuracy) * 100
        print(f"✓ Agent方法表现更好，准确率提升 {improvement:.2f}%")
    elif agent_metrics.accuracy < fixed_metrics.accuracy:
        decline = (fixed_metrics.accuracy - agent_metrics.accuracy) * 100
        print(f"⚠️  固定方法表现更好，Agent方法准确率降低 {decline:.2f}%")
    else:
        print(f"两种方法表现相同")
    
    # 分歧分析
    analyze_disagreements(agent_results, labels)
    
    # 5. 保存详细报告
    if args.output:
        output_file = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"comparison_report_{timestamp}.json"
    
    save_comparison_report(
        agent_metrics, fixed_metrics,
        agent_predictions, fixed_predictions,
        labels, output_file
    )
    
    print("\n" + "=" * 70)
    print("对比完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

