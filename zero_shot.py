#!/usr/bin/env python3
"""
Zero-shot evaluation of LLMs for CPC multi-label patent classification.

This script implements the paper's ZERO-SHOT (clean) setting:
- No few-shot examples
- No retrieval / no RAG
- Deterministic decoding (do_sample=False)
- Output constraint: 1 to 7 CPC subclasses (4-character level)
- Post-processing:
  - Parse strict JSON {"labels": [...]} (with a fallback to extracting the first JSON object)
  - Normalize to 4-character subclasses
  - Keep at most 7 labels
  - If parsing returns empty, apply a minimal non-empty fallback consistent with the "1..7" requirement:
    take the first CPC-like code found in the generation.

Inputs:
- A TSV test file with columns: title, abstract, labels

Outputs (per model) under: <output_dir>/<model_name>/
- metrics.json
- predictions.jsonl
- per_label_section.csv
- per_label_class.csv
- per_label_subclass.csv
- codecarbon emissions file(s) (if codecarbon is installed and enabled)
"""

import os
import re
import json
import ast
import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


SYSTEM_PROMPT = (
    "You are an expert patent examiner for the Cooperative Patent "
    "Classification (CPC) system. "
    "Your goal is HIGH RECALL classification: it is worse to miss a relevant "
    "CPC subclass "
    "than to include a marginally relevant one. "
    "You assign MULTIPLE CPC subclasses (multi-label) to each patent. "
    "Return ONLY a strict JSON object with a single key \"labels\", "
    "whose value is a list of CPC subclass codes (e.g. [\"G06Q\", \"Y02D\"]). "
    "Return JSON ONLY (no extra text)."
)

USER_PROMPT_TEMPLATE = (
    "Classify the following patent into CPC subclasses (4-character level "
    "like A01B, G06F, Y02D).\n\n"
    "PATENT TEXT:\n"
    "----------------------------------------\n"
    "{patent_text}\n"
    "----------------------------------------\n"
    "TASK:\n"
    "- Assign ALL relevant CPC subclasses (multi-label).\n"
    "- Output between 1 and 7 CPC subclass codes.\n"
    "- Do NOT invent codes that are not real CPC subclasses.\n"
    "OUTPUT FORMAT (STRICT):\n"
    "{\"labels\": [\"G06F\", \"H04L\"]}\n"
)

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_CPC_TOKEN_RE = re.compile(r"\b[A-HY]\d{2}[A-Z]\b")


def normalize_cpc_label(code: str) -> str:
    if not code:
        return ""
    c = str(code).strip().upper()
    if not c:
        return ""
    c = c.split("/")[0]
    if len(c) >= 4 and c[0].isalpha() and c[1:3].isdigit():
        return c[:4]
    return c[:4]


def parse_labels(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        return [p.strip() for p in s.split(";") if p.strip()]
    return [str(raw).strip()]


def build_text(title: str, abstract: str) -> str:
    return ((title or "").strip() + ". " + (abstract or "").strip()).strip()


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_predicted_labels(text: str, max_labels: int = 7) -> List[str]:
    if not text:
        return []

    def postprocess(labs: List[str]) -> List[str]:
        norm = [normalize_cpc_label(x) for x in labs]
        norm = [x for x in norm if x]
        norm = _dedupe_keep_order(norm)
        return norm[:max_labels]

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            labs = obj.get("labels", [])
            if isinstance(labs, list):
                return postprocess([str(x) for x in labs])
    except Exception:
        pass

    m = _JSON_OBJ_RE.search(text)
    if m:
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                labs = obj.get("labels", [])
                if isinstance(labs, list):
                    return postprocess([str(x) for x in labs])
        except Exception:
            pass

    return []


def enforce_nonempty(pred: List[str], raw_text: str, max_labels: int = 7) -> List[str]:
    if pred:
        return pred[:max_labels]
    cands = _CPC_TOKEN_RE.findall((raw_text or "").upper())
    for c in cands:
        c = normalize_cpc_label(c)
        if c:
            return [c]
    return []


def _micro_prf_from_sets(true_sets: List[set], pred_sets: List[set]) -> Tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0
    for t, p in zip(true_sets, pred_sets):
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def _labels_to_level_set(labels: List[str], level: str) -> set:
    out = set()
    for lab in labels or []:
        lab = normalize_cpc_label(lab)
        if not lab:
            continue
        if level == "section":
            out.add(lab[0])
        elif level == "class":
            out.add(lab[:3])
        elif level == "subclass":
            out.add(lab[:4])
        else:
            raise ValueError(level)
    return out


def compute_level_micro_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in ("section", "class", "subclass"):
        true_sets = [_labels_to_level_set(labs, level) for labs in y_true]
        pred_sets = [_labels_to_level_set(labs, level) for labs in y_pred]
        p, r, f1, tp, fp, fn = _micro_prf_from_sets(true_sets, pred_sets)
        out[level] = {
            "micro_precision": float(p),
            "micro_recall": float(r),
            "micro_f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    return out


def cpc_hierarchy_tags(label: str) -> List[str]:
    if not label or not isinstance(label, str):
        return []
    main = label.strip().split("/")[0]
    tags: List[str] = []
    if len(main) >= 1 and main[0].isalpha():
        tags.append(f"SECTION_{main[0]}")
    if len(main) >= 3 and main[0].isalpha() and main[1:3].isdigit():
        tags.append(f"CLASS_{main[0:3]}")
    if len(main) >= 4:
        tags.append(f"SUBCLASS_{main[0:4]}")
    return tags


def build_hier_sets(labels: List[str]) -> List[str]:
    tags: List[str] = []
    for lab in labels or []:
        tags.extend(cpc_hierarchy_tags(lab))
    return sorted(set(tags))


def compute_hierarchical_micro_f1(y_true: List[List[str]], y_pred: List[List[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for t_labs, p_labs in zip(y_true, y_pred):
        t_tags = set(build_hier_sets(t_labs))
        p_tags = set(build_hier_sets(p_labs))
        tp += len(t_tags & p_tags)
        fp += len(p_tags - t_tags)
        fn += len(t_tags - p_tags)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def flat_acc_at_1(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    correct = 0
    total = 0
    for t, p in zip(y_true, y_pred):
        if not p:
            continue
        total += 1
        if p[0] in set(t):
            correct += 1
    return correct / total if total else 0.0


def label_count_diagnostics(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    gold_counts = [len(x) for x in gold]
    pred_counts = [len(x) for x in pred]
    empty_rate = float(np.mean([1.0 if len(x) == 0 else 0.0 for x in pred])) if pred else 0.0
    return {
        "avg_gold_labels_per_patent": float(np.mean(gold_counts)) if gold_counts else 0.0,
        "avg_pred_labels_per_patent": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "empty_prediction_rate": empty_rate,
    }


def _levelize_labels(labels: List[str], level: str) -> List[str]:
    out: List[str] = []
    for lab in labels or []:
        lab = normalize_cpc_label(lab)
        if not lab:
            continue
        if level == "section":
            out.append(lab[0])
        elif level == "class":
            out.append(lab[:3])
        elif level == "subclass":
            out.append(lab[:4])
        else:
            raise ValueError(level)
    return _dedupe_keep_order(out)


def per_label_table(
    y_true_level: List[List[str]],
    y_pred_level: List[List[str]],
    classes: List[str],
) -> List[Dict[str, float]]:
    mlb = MultiLabelBinarizer(classes=classes)
    Yt = mlb.fit_transform(y_true_level)
    Yp = mlb.transform(y_pred_level)

    Yt_b = (Yt == 1)
    Yp_b = (Yp == 1)

    tp = np.logical_and(Yt_b, Yp_b).sum(axis=0)
    fp = np.logical_and(~Yt_b, Yp_b).sum(axis=0)
    fn = np.logical_and(Yt_b, ~Yp_b).sum(axis=0)
    tn = np.logical_and(~Yt_b, ~Yp_b).sum(axis=0)

    rows: List[Dict[str, float]] = []
    for j, lab in enumerate(classes):
        tpj, fpj, fnj, tnj = int(tp[j]), int(fp[j]), int(fn[j]), int(tn[j])
        prec = tpj / (tpj + fpj) if (tpj + fpj) else 0.0
        rec = tpj / (tpj + fnj) if (tpj + fnj) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        acc = (tpj + tnj) / (tpj + tnj + fpj + fnj) if (tpj + tnj + fpj + fnj) else 0.0
        rows.append(
            {
                "label": lab,
                "support": float(tpj + fnj),
                "pred_pos": float(tpj + fpj),
                "tp": float(tpj),
                "fp": float(fpj),
                "fn": float(fnj),
                "tn": float(tnj),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
            }
        )
    rows.sort(key=lambda r: (r["support"], r["f1"]), reverse=True)
    return rows


def save_per_label_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_llm(model_path: Path, local_only: bool):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model directory: {model_path}")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tok = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        device_map="auto",
        torch_dtype=dtype,
    ).eval()

    return mdl, tok


def generate_batch(
    model,
    tokenizer,
    titles: List[str],
    abstracts: List[str],
    max_new_tokens: int,
    max_labels: int,
) -> Tuple[List[List[str]], List[str]]:
    prompts: List[str] = []
    for title, abstract in zip(titles, abstracts):
        patent_text = build_text(title, abstract)
        user_prompt = USER_PROMPT_TEMPLATE.format(patent_text=patent_text)
        prompts.append(build_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    preds: List[List[str]] = []
    raw_texts: List[str] = []

    for i in range(len(prompts)):
        prompt_len = int(inputs["attention_mask"][i].sum().item())
        gen_ids = outputs[i][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw_texts.append(gen_text)

        labs = parse_predicted_labels(gen_text, max_labels=max_labels)
        labs = enforce_nonempty(labs, gen_text, max_labels=max_labels)
        preds.append(labs)

    return preds, raw_texts


def evaluate_model(
    model_name: str,
    model_path: Path,
    test_path: Path,
    out_root: Path,
    title_col: str,
    abstract_col: str,
    labels_col: str,
    batch_size: int,
    max_new_tokens: int,
    max_eval: Optional[int],
    max_labels: int,
    local_only: bool,
    codecarbon: bool,
) -> None:
    outdir = out_root / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("csv", data_files={"test": str(test_path)}, delimiter="\t")["test"]
    if max_eval is not None:
        raw = raw.select(range(min(int(max_eval), len(raw))))

    titles: List[str] = []
    abstracts: List[str] = []
    gold_labels: List[List[str]] = []

    for ex in raw:
        titles.append(ex.get(title_col, "") or "")
        abstracts.append(ex.get(abstract_col, "") or "")
        labs = [normalize_cpc_label(x) for x in parse_labels(ex.get(labels_col))]
        labs = [x for x in labs if x]
        gold_labels.append(labs)

    all_gold_labels = sorted(set(l for labs in gold_labels for l in labs))
    if not all_gold_labels:
        raise ValueError("No gold labels found after parsing test labels.")

    model, tok = load_llm(model_path, local_only=local_only)

    tracker = None
    if codecarbon and EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name=f"zero_shot_clean_{model_name}",
            output_dir=str(outdir),
            measure_power_secs=10,
            save_to_file=True,
            log_level="error",
        )

    start_time = time.time()
    if tracker is not None:
        tracker.start()

    pred_labels: List[List[str]] = []
    raw_generations: List[str] = []

    for i in tqdm(range(0, len(titles), batch_size), desc=f"Inference [{model_name}]"):
        bt = titles[i : i + batch_size]
        ba = abstracts[i : i + batch_size]
        batch_preds, batch_raw = generate_batch(
            model=model,
            tokenizer=tok,
            titles=bt,
            abstracts=ba,
            max_new_tokens=max_new_tokens,
            max_labels=max_labels,
        )
        pred_labels.extend(batch_preds)
        raw_generations.extend(batch_raw)

    emissions_kg = tracker.stop() if tracker is not None else None
    total_time_sec = time.time() - start_time

    mlb = MultiLabelBinarizer(classes=all_gold_labels)
    y_true_bin = mlb.fit_transform(gold_labels)
    y_pred_bin = mlb.transform(pred_labels)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )

    flat_acc1 = flat_acc_at_1(gold_labels, pred_labels)
    h_p, h_r, h_f1 = compute_hierarchical_micro_f1(gold_labels, pred_labels)
    diag = label_count_diagnostics(gold_labels, pred_labels)
    level_metrics = compute_level_micro_metrics(gold_labels, pred_labels)

    gold_section = [_levelize_labels(l, "section") for l in gold_labels]
    pred_section = [_levelize_labels(l, "section") for l in pred_labels]
    gold_class = [_levelize_labels(l, "class") for l in gold_labels]
    pred_class = [_levelize_labels(l, "class") for l in pred_labels]
    gold_subclass = [_levelize_labels(l, "subclass") for l in gold_labels]
    pred_subclass = [_levelize_labels(l, "subclass") for l in pred_labels]

    section_universe = sorted(set(x for labs in gold_section for x in labs))
    class_universe = sorted(set(x for labs in gold_class for x in labs))
    subclass_universe = sorted(set(x for labs in gold_subclass for x in labs))

    save_per_label_csv(per_label_table(gold_section, pred_section, section_universe), outdir / "per_label_section.csv")
    save_per_label_csv(per_label_table(gold_class, pred_class, class_universe), outdir / "per_label_class.csv")
    save_per_label_csv(per_label_table(gold_subclass, pred_subclass, subclass_universe), outdir / "per_label_subclass.csv")

    metrics = {
        "model_name": model_name,
        "num_examples": len(gold_labels),
        "total_inference_time_sec": float(total_time_sec),
        "avg_time_per_patent_sec": float(total_time_sec / len(gold_labels)) if gold_labels else None,
        "total_co2_emissions_kg": float(emissions_kg) if emissions_kg is not None else None,
        "label_count_diagnostics": diag,
        "flat_multilabel_subclass": {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "acc_at_1": float(flat_acc1),
        },
        "hierarchical_micro_tags": {"precision": float(h_p), "recall": float(h_r), "f1": float(h_f1)},
        "hierarchical_levels": level_metrics,
        "config": {
            "batch_size": int(batch_size),
            "max_new_tokens": int(max_new_tokens),
            "max_eval": int(max_eval) if max_eval is not None else None,
            "max_labels": int(max_labels),
            "decoding": {"do_sample": False},
            "prompt": {
                "system_prompt": "paper_zero_shot_system_prompt",
                "user_prompt": "paper_zero_shot_user_prompt",
            },
        },
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, (t, a, gold, pred, rawtxt) in enumerate(
            zip(titles, abstracts, gold_labels, pred_labels, raw_generations)
        ):
            rec = {
                "idx": idx,
                "title": t,
                "abstract": a,
                "gold_labels": gold,
                "pred_labels": pred,
                "raw_generation": rawtxt,
            }
            f.write(json.dumps(rec) + "\n")

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_path", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--title_col", type=str, default="title")
    ap.add_argument("--abstract_col", type=str, default="abstract")
    ap.add_argument("--labels_col", type=str, default="labels")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--max_labels", type=int, default=7)

    ap.add_argument("--local_only", action="store_true", default=True)
    ap.add_argument("--use_codecarbon", action="store_true", default=True)

    ap.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "MODEL_PATH"),
        required=True,
        help="Repeatable. Example: --model qwen /path/to/Qwen2.5-7B-Instruct",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not args.test_path.is_file():
        raise FileNotFoundError(f"Missing test file: {args.test_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_path_str in args.model:
        evaluate_model(
            model_name=model_name,
            model_path=Path(model_path_str),
            test_path=args.test_path,
            out_root=args.output_dir,
            title_col=args.title_col,
            abstract_col=args.abstract_col,
            labels_col=args.labels_col,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_eval=args.max_eval,
            max_labels=args.max_labels,
            local_only=args.local_only,
            codecarbon=args.use_codecarbon,
        )


if __name__ == "__main__":
    main()
