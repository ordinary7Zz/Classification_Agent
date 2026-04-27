import os
import json
import csv
import argparse
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ----------------------------
# Utils (same as MedGemma script)
# ----------------------------
def load_labels(label_json: str) -> List[Dict]:
    """
    Expected label format (list of dict):
      [{"filename": "...", "malignancy": 0/1}, ...]
    """
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Label JSON must be a list of records.")
    for r in data:
        if "filename" not in r or "malignancy" not in r:
            raise ValueError("Each record must contain keys: filename, malignancy.")
        r["malignancy"] = int(r["malignancy"])
        if r["malignancy"] not in (0, 1):
            raise ValueError(f"malignancy must be 0/1, got {r['malignancy']}")
    return data


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    # ultrasound images might be grayscale; convert to RGB to be safe
    return img.convert("RGB")


@torch.inference_mode()
def score_string_logprob(
    model: Qwen3VLForConditionalGeneration,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    target_text: str,
) -> float:
    """
    Compute log P(target_text | base_inputs) using cached KV for efficiency.

    Steps:
      1) Forward pass on base inputs -> get logits for next token + past_key_values
      2) For each token in target tokens:
          add logprob of token given current logits
          feed token with past_key_values to get next logits
    """
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(tgt_ids) == 0:
        raise ValueError(f"target_text tokenized to empty sequence: {repr(target_text)}")

    out = model(**base_inputs, use_cache=True)
    logits = out.logits
    past = out.past_key_values

    next_logits = logits[:, -1, :]  # [1, V]
    logprob = 0.0

    for tid in tgt_ids:
        step_logprob = torch.log_softmax(next_logits, dim=-1)[0, tid].item()
        logprob += step_logprob

        input_ids = torch.tensor([[tid]], device=next_logits.device, dtype=torch.long)
        out2 = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        next_logits = out2.logits[:, -1, :]
        past = out2.past_key_values

    return float(logprob)


def build_qwen3_vl_inputs(processor: AutoProcessor, img: Image.Image) -> Dict[str, torch.Tensor]:
    """
    Build Qwen3-VL multimodal chat inputs. We force output to be exactly one character: 0 or 1.
    """
    system_text = (
        "You are a medical imaging assistant. "
        "You will be given a thyroid ultrasound image. "
        "Your task is binary malignancy classification."
    )

    user_text = (
        "Task: Thyroid ultrasound nodule malignancy classification.\n"
        "Output exactly one character: 0 or 1.\n"
        "0 = benign, 1 = malignant.\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]},
    ]

    # Qwen3-VL processor can directly build multimodal tensors from chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def prob_malignant_from_logits(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    img: Image.Image,
) -> Tuple[float, Dict[str, float]]:
    """
    Return P(malignant=1) by comparing log-probabilities of "0" vs "1" candidates.

    Because tokenization can differ (leading space/newline), we try a few safe candidate forms.
    For each pair (c0, c1), compute:
        p1 = exp(lp1) / (exp(lp0) + exp(lp1))
    Use the first pair that works.
    """
    tokenizer = processor.tokenizer
    device = model.device

    base_inputs = build_qwen3_vl_inputs(processor, img)
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}

    candidate_pairs = [
        (" 0", " 1"),
        ("0", "1"),
        ("\n0", "\n1"),
        (" 0\n", " 1\n"),
    ]

    last_err = None
    for c0, c1 in candidate_pairs:
        try:
            lp0 = score_string_logprob(model, tokenizer, base_inputs, c0)
            lp1 = score_string_logprob(model, tokenizer, base_inputs, c1)

            m = max(lp0, lp1)
            p0 = torch.exp(torch.tensor(lp0 - m)).item()
            p1 = torch.exp(torch.tensor(lp1 - m)).item()
            p1_norm = p1 / (p0 + p1 + 1e-12)

            return float(p1_norm), {"logp_0": lp0, "logp_1": lp1, "cand0": c0, "cand1": c1}
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All candidate pairs failed. Last error: {last_err}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/wangbd8/workspace/qwen3_vl_8b_instruct",
        help="Local path to Qwen3-VL-8B-Instruct model dir",
    )
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Directory containing images referenced by filename in label json")
    ap.add_argument("--label_json", type=str, required=True,
                    help="Label json path, e.g. /mnt/data/tn3k_test_label.json")
    ap.add_argument("--out_csv", type=str, default="qwen3_vl_8b_preds.csv",
                    help="Output CSV with per-image predictions")
    ap.add_argument("--use_fast_processor", action="store_true",
                    help="Use fast processor (default: slow for stability)")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="Computation dtype on GPU")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold on P(malignant)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Run only first N samples for quick test (-1 for all)")
    ap.add_argument("--attn_impl", type=str, default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                    help="Attention implementation; use flash_attention_2 if installed")
    args = ap.parse_args()

    labels = load_labels(args.label_json)
    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    # dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # load processor/model
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=args.use_fast_processor)

    model_kwargs = dict(
        device_map="cuda",
        torch_dtype=torch_dtype,
    )
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, **model_kwargs)
    model.eval()

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []

    missing_files = 0
    bad_images = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gt_malignant", "p_malignant", "pred_malignant", "logp_0", "logp_1", "cand0", "cand1"])

        for r in tqdm(labels, desc="Infer"):
            fn = r["filename"]
            gt = int(r["malignancy"])

            img_path = os.path.join(args.image_dir, fn)
            if not os.path.exists(img_path):
                missing_files += 1
                continue

            try:
                img = safe_open_image(img_path)
            except Exception:
                bad_images += 1
                continue

            p1, dbg = prob_malignant_from_logits(model=model, processor=processor, img=img)
            pred = 1 if p1 >= args.threshold else 0

            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)

            w.writerow([fn, gt, f"{p1:.6f}", pred, f"{dbg['logp_0']:.6f}", f"{dbg['logp_1']:.6f}", dbg["cand0"], dbg["cand1"]])

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated (all missing/bad?). Check paths.")

    # Metrics (positive class = malignant=1)
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # = sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    print("\n==== Qwen3-VL-8B-Instruct Thyroid Binary Classification Metrics (pos=malignant=1) ====")
    print(f"Model dir:         {args.model_dir}")
    print(f"Evaluated samples: {len(y_true)}")
    print(f"Missing files:     {missing_files}")
    print(f"Bad images:        {bad_images}")
    print(f"AUROC:             {auroc:.6f}")
    print(f"AUPRC:             {auprc:.6f}")
    print(f"Acc:               {acc:.6f}")
    print(f"Prec:              {prec:.6f}")
    print(f"Recall:            {rec:.6f}")
    print(f"F1:                {f1:.6f}")
    print(f"Sensitivity:       {sensitivity:.6f}")
    print(f"Specificity:       {specificity:.6f}")
    print(f"Confusion (tn fp fn tp): {tn} {fp} {fn} {tp}")
    print(f"Saved per-image predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()

