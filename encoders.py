#!/usr/bin/env python3
"""
Evaluate saved BERT-family multi-label classifiers under two conditions:

1) Supervised encoder baseline (full label space)
2) Supervised encoder + allowed-set masking (Top-K) via E5 retrieval over CPC definitions

Outputs are saved under:
  <output_base_dir>/<run_name>/eval_outputs/seed_<seed>/
"""

import os
import json
import ast
import time
import csv
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from codecarbon import EmissionsTracker


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass


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
        parts = s.split(";")
        return [p.strip() for p in parts if p.strip()]
    return [str(raw).strip()]


def build_text(title: str, abstract: str) -> str:
    return ((title or "").strip() + ". " + (abstract or "").strip()).strip()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_label_list_from_label_mapping(mapping_path: Path) -> List[str]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")

    obj = json.loads(mapping_path.read_text(encoding="utf-8"))

    if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
        labs = obj["labels"]
    elif isinstance(obj, dict) and "id2label" in obj:
        id2label = obj["id2label"]
        labs = [id2label[str(i)] for i in sorted(map(int, id2label.keys()))]
    elif isinstance(obj, dict) and "label2id" in obj:
        label2id = obj["label2id"]
        labs = [lab for lab, _ in sorted(label2id.items(), key=lambda kv: int(kv[1]))]
    elif isinstance(obj, list):
        labs = obj
    elif isinstance(obj, dict) and all(str(k).isdigit() for k in obj.keys()):
        labs = [obj[str(i)] for i in sorted(map(int, obj.keys()))]
    else:
        raise ValueError(
            f"Unrecognized label mapping format in {mapping_path}. Keys: {list(obj)[:10]}"
        )

    labs = [normalize_cpc_label(x) for x in labs]
    labs = [x for x in labs if x]
    if not labs:
        raise ValueError(f"No valid labels parsed from {mapping_path}")
    return labs


def resolve_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = list(run_dir.glob("checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* dirs found in {run_dir}")

    def step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    ckpts = sorted(ckpts, key=step)
    return ckpts[-1]


def load_split(
    tsv_path: Path,
    title_col: str,
    abstract_col: str,
    labels_col: str,
) -> Tuple[List[str], List[List[str]]]:
    ds = load_dataset("csv", data_files={"d": str(tsv_path)}, delimiter="\t")["d"]
    texts, labels = [], []
    for ex in ds:
        t = ex.get(title_col, "") or ""
        a = ex.get(abstract_col, "") or ""
        texts.append(build_text(t, a))
        labs = [normalize_cpc_label(x) for x in parse_labels(ex.get(labels_col))]
        labs = [x for x in labs if x]
        labels.append(labs)
    return texts, labels


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def make_dataloader(
    texts: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    ds = TextDataset(texts)

    def collate(batch_texts: List[str]):
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, batch_texts

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def forward_probs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    dl = make_dataloader(texts, tokenizer, batch_size, max_length)
    device = next(model.parameters()).device
    all_logits = []
    for enc, _ in tqdm(dl, desc="Forward", leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        all_logits.append(out.logits.detach().cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    return sigmoid(logits)


def decode_probs_to_labels(
    probs: np.ndarray,
    label_list: List[str],
    threshold: Optional[float],
    min_labels: int = 1,
    max_labels: int = 7,
) -> List[List[str]]:
    out: List[List[str]] = []
    for row in probs:
        pairs = list(zip(label_list, row.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if threshold is None:
            chosen = [lab for lab, _ in pairs[:max_labels]]
        else:
            chosen = [lab for lab, p in pairs if p >= threshold]
            if len(chosen) < min_labels:
                chosen = [pairs[0][0]] if pairs else []
            if len(chosen) > max_labels:
                chosen = [lab for lab, _ in pairs[:max_labels]]

        out.append(chosen)
    return out


def micro_metrics_from_sets(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = set(t), set(p)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def calibrate_threshold_on_dev(
    dev_probs: np.ndarray,
    dev_gold: List[List[str]],
    label_list: List[str],
    thresh_grid: np.ndarray,
    min_labels: int,
    max_labels: int,
) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in thresh_grid:
        pred = decode_probs_to_labels(
            dev_probs,
            label_list,
            threshold=float(t),
            min_labels=min_labels,
            max_labels=max_labels,
        )
        _, _, f1 = micro_metrics_from_sets(dev_gold, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def flat_acc_at_1(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    correct = 0
    total = 0
    for t, p in zip(y_true, y_pred):
        if not p:
            continue
        total += 1
        if p[0] in set(t):
            correct += 1
    return correct / total if total > 0 else 0.0


def label_count_diagnostics(
    gold: List[List[str]],
    pred: List[List[str]],
) -> Dict[str, float]:
    gold_counts = [len(x) for x in gold]
    pred_counts = [len(x) for x in pred]
    empty_rate = float(np.mean([1.0 if len(x) == 0 else 0.0 for x in pred])) if pred else 0.0
    return {
        "avg_gold_labels_per_patent": float(np.mean(gold_counts)) if gold_counts else 0.0,
        "avg_pred_labels_per_patent": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "empty_prediction_rate": empty_rate,
    }


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


def compute_level_micro_metrics(
    y_true_labels: List[List[str]],
    y_pred_labels: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    def _micro_prf(true_sets: List[set], pred_sets: List[set]) -> Tuple[float, float, float, int, int, int]:
        tp = fp = fn = 0
        for t, p in zip(true_sets, pred_sets):
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, tp, fp, fn

    out = {}
    for level in ["section", "class", "subclass"]:
        true_sets = [_labels_to_level_set(labs, level) for labs in y_true_labels]
        pred_sets = [_labels_to_level_set(labs, level) for labs in y_pred_labels]
        p, r, f1, tp, fp, fn = _micro_prf(true_sets, pred_sets)
        out[level] = {
            "micro_precision": float(p),
            "micro_recall": float(r),
            "micro_f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    return out


def per_label_table(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    classes: List[str],
) -> List[Dict[str, float]]:
    mlb = MultiLabelBinarizer(classes=classes)
    Yt = mlb.fit_transform(y_true)
    Yp = mlb.transform(y_pred)

    Yt_b = (Yt == 1)
    Yp_b = (Yp == 1)

    tp = np.logical_and(Yt_b, Yp_b).sum(axis=0)
    fp = np.logical_and(~Yt_b, Yp_b).sum(axis=0)
    fn = np.logical_and(Yt_b, ~Yp_b).sum(axis=0)
    tn = np.logical_and(~Yt_b, ~Yp_b).sum(axis=0)

    rows: List[Dict[str, float]] = []
    for j, lab in enumerate(classes):
        tpj, fpj, fnj, tnj = int(tp[j]), int(fp[j]), int(fn[j]), int(tn[j])
        prec = tpj / (tpj + fpj) if (tpj + fpj) > 0 else 0.0
        rec = tpj / (tpj + fnj) if (tpj + fnj) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tpj + tnj) / (tpj + tnj + fpj + fnj) if (tpj + tnj + fpj + fnj) > 0 else 0.0
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


def load_e5_encoder(embed_model_path: Path):
    from transformers import AutoTokenizer as ETok
    from transformers import AutoModel as EModel

    if not embed_model_path.is_dir():
        raise FileNotFoundError(f"Missing embedding model dir: {embed_model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ETok.from_pretrained(str(embed_model_path), local_files_only=True)
    mdl = EModel.from_pretrained(str(embed_model_path), local_files_only=True).to(device)
    mdl.eval()
    return mdl, tok, device


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts_e5(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    prefix: str,
    max_length: int = 512,
) -> torch.Tensor:
    if prefix:
        pfx = prefix.strip() + ": "
        texts = [pfx + (t or "").strip() for t in texts]
    else:
        texts = [(t or "").strip() for t in texts]

    outs = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i : i + batch_size]
        enc = tokenizer(
            bt,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        outs.append(emb.cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def retrieve_allowed_topk(
    patent_texts: List[str],
    k: int,
    e5_model,
    e5_tokenizer,
    e5_device: str,
    label_codes: List[str],
    label_embeds_cpu: torch.Tensor,
    patent_emb_batch: int,
) -> List[List[str]]:
    q_emb_cpu = encode_texts_e5(
        patent_texts,
        e5_model,
        e5_tokenizer,
        e5_device,
        batch_size=min(patent_emb_batch, max(1, len(patent_texts))),
        prefix="query",
    )
    q = q_emb_cpu.to(e5_device)
    L = label_embeds_cpu.to(e5_device)
    sims = q @ L.T
    top_idx = torch.topk(sims, k=min(k, L.size(0)), dim=1).indices.cpu().tolist()
    return [[label_codes[j] for j in idxs] for idxs in top_idx]


def apply_allowed_mask(
    probs: np.ndarray,
    label_list: List[str],
    allowed_lists: List[List[str]],
) -> np.ndarray:
    lab2i = {lab: i for i, lab in enumerate(label_list)}
    masked = np.zeros_like(probs)
    for i, allowed in enumerate(allowed_lists):
        idxs = [lab2i[a] for a in allowed if a in lab2i]
        if idxs:
            masked[i, idxs] = probs[i, idxs]
    return masked


def evaluate_one_model(
    model_cfg: Dict[str, str],
    label_list: List[str],
    seed: int,
    output_base_dir: Path,
    dev_path: Path,
    test_path: Path,
    title_col: str,
    abstract_col: str,
    labels_col: str,
    max_length: int,
    eval_batch_size: int,
    calibrate_threshold_on_dev: bool,
    thresh_grid: np.ndarray,
    min_labels: int,
    max_labels: int,
    save_predictions: bool,
    embed_model_path: Path,
    top_k_cpc: int,
    cpc_emb_batch: int,
    patent_emb_batch: int,
    cpc_labels: Dict[str, str],
) -> None:
    set_all_seeds(seed)

    run_name = model_cfg["run_name"]
    model_name = model_cfg["name"]
    base_tok_dir = Path(model_cfg["base_tokenizer_dir"])

    run_dir = output_base_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    ckpt_dir = resolve_latest_checkpoint(run_dir)
    out_dir = ensure_dir(run_dir / "eval_outputs" / f"seed_{seed}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_tok_dir), local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        num_labels=len(label_list),
        problem_type="multi_label_classification",
    )

    model.config.id2label = {i: lab for i, lab in enumerate(label_list)}
    model.config.label2id = {lab: i for i, lab in enumerate(label_list)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    dev_texts, dev_gold = load_split(dev_path, title_col, abstract_col, labels_col)
    test_texts, test_gold = load_split(test_path, title_col, abstract_col, labels_col)

    tracker = EmissionsTracker(
        project_name=f"encoder_eval_{model_name}_seed{seed}",
        output_dir=str(out_dir),
        measure_power_secs=10,
        save_to_file=True,
        log_level="error",
    )
    tracker.start()
    t0 = time.time()
    dev_probs = forward_probs(model, tokenizer, dev_texts, eval_batch_size, max_length)
    test_probs = forward_probs(model, tokenizer, test_texts, eval_batch_size, max_length)
    total_time = time.time() - t0
    emissions_kg = tracker.stop()

    if calibrate_threshold_on_dev:
        thr = calibrate_threshold_on_dev_fn(
            dev_probs=dev_probs,
            dev_gold=dev_gold,
            label_list=label_list,
            thresh_grid=thresh_grid,
            min_labels=min_labels,
            max_labels=max_labels,
        )
    else:
        thr = None

    test_universe = sorted(set(l for labs in test_gold for l in labs))
    if not test_universe:
        raise ValueError("No gold labels on test after parsing.")
    mlb = MultiLabelBinarizer(classes=test_universe)
    y_true_bin = mlb.fit_transform(test_gold)

    test_pred_full = decode_probs_to_labels(
        test_probs,
        label_list,
        threshold=thr,
        min_labels=min_labels,
        max_labels=max_labels,
    )
    y_pred_bin_full = mlb.transform(test_pred_full)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin_full, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin_full, average="macro", zero_division=0
    )
    acc1 = flat_acc_at_1(test_gold, test_pred_full)
    diag = label_count_diagnostics(test_gold, test_pred_full)
    level_metrics = compute_level_micro_metrics(test_gold, test_pred_full)

    metrics_full = {
        "seed": int(seed),
        "model": model_name,
        "run_name": run_name,
        "checkpoint": str(ckpt_dir),
        "condition": "full_label_space",
        "num_test": len(test_gold),
        "decode_threshold_dev": thr,
        "min_labels": min_labels,
        "max_labels": max_labels,
        "total_forward_time_sec_dev_plus_test": float(total_time),
        "total_emissions_kg": float(emissions_kg) if emissions_kg is not None else None,
        "flat_multilabel_subclass": {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "acc_at_1": float(acc1),
        },
        "label_count_diagnostics": diag,
        "hierarchical_levels": level_metrics,
    }
    (out_dir / "metrics_full.json").write_text(json.dumps(metrics_full, indent=2), encoding="utf-8")

    rows_full = per_label_table(test_gold, test_pred_full, classes=test_universe)
    save_per_label_csv(rows_full, out_dir / "per_label_subclass_full.csv")

    if save_predictions:
        with (out_dir / "predictions_full.jsonl").open("w", encoding="utf-8") as f:
            for i, (txt, g, p) in enumerate(zip(test_texts, test_gold, test_pred_full)):
                f.write(json.dumps({"idx": i, "text": txt, "gold": g, "pred": p}) + "\n")

    if not cpc_labels:
        raise ValueError("CPC definitions dict is empty.")

    e5_model, e5_tok, e5_dev = load_e5_encoder(embed_model_path)

    label_set = set(label_list)
    label_codes_all = sorted({normalize_cpc_label(k) for k in cpc_labels.keys() if normalize_cpc_label(k)})
    label_codes_all = [c for c in label_codes_all if c in label_set]
    if not label_codes_all:
        raise ValueError("No overlap between CPC definition keys and classifier label list.")

    label_texts = [f"{c}: {cpc_labels.get(c, '')}" for c in label_codes_all]
    label_embeds_cpu = encode_texts_e5(
        label_texts,
        e5_model,
        e5_tok,
        e5_dev,
        batch_size=cpc_emb_batch,
        prefix="passage",
    ).cpu()

    allowed_lists: List[List[str]] = []
    for i in tqdm(range(0, len(test_texts), eval_batch_size), desc="Retrieving allowed top-K"):
        bt = test_texts[i : i + eval_batch_size]
        allowed_lists.extend(
            retrieve_allowed_topk(
                bt,
                top_k_cpc,
                e5_model,
                e5_tok,
                e5_dev,
                label_codes_all,
                label_embeds_cpu,
                patent_emb_batch=patent_emb_batch,
            )
        )

    test_probs_masked = apply_allowed_mask(test_probs, label_list, allowed_lists)
    test_pred_masked = decode_probs_to_labels(
        test_probs_masked,
        label_list,
        threshold=thr,
        min_labels=min_labels,
        max_labels=max_labels,
    )

    y_pred_bin_masked = mlb.transform(test_pred_masked)
    micro_p_m, micro_r_m, micro_f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin_masked, average="micro", zero_division=0
    )
    macro_p_m, macro_r_m, macro_f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin_masked, average="macro", zero_division=0
    )
    acc1_m = flat_acc_at_1(test_gold, test_pred_masked)
    diag_m = label_count_diagnostics(test_gold, test_pred_masked)
    level_metrics_m = compute_level_micro_metrics(test_gold, test_pred_masked)

    metrics_masked = {
        "seed": int(seed),
        "model": model_name,
        "run_name": run_name,
        "checkpoint": str(ckpt_dir),
        "condition": f"allowed_set_mask_topk{top_k_cpc}",
        "num_test": len(test_gold),
        "decode_threshold_dev": thr,
        "min_labels": min_labels,
        "max_labels": max_labels,
        "top_k": top_k_cpc,
        "allowed_label_source": "E5 retrieval over CPC definitions",
        "flat_multilabel_subclass": {
            "micro_precision": float(micro_p_m),
            "micro_recall": float(micro_r_m),
            "micro_f1": float(micro_f1_m),
            "macro_precision": float(macro_p_m),
            "macro_recall": float(macro_r_m),
            "macro_f1": float(macro_f1_m),
            "acc_at_1": float(acc1_m),
        },
        "label_count_diagnostics": diag_m,
        "hierarchical_levels": level_metrics_m,
    }
    (out_dir / f"metrics_masked_topk{top_k_cpc}.json").write_text(
        json.dumps(metrics_masked, indent=2), encoding="utf-8"
    )

    rows_masked = per_label_table(test_gold, test_pred_masked, classes=test_universe)
    save_per_label_csv(rows_masked, out_dir / f"per_label_subclass_masked_topk{top_k_cpc}.csv")

    if save_predictions:
        with (out_dir / f"predictions_masked_topk{top_k_cpc}.jsonl").open("w", encoding="utf-8") as f:
            for i, (txt, g, p, alw) in enumerate(zip(test_texts, test_gold, test_pred_masked, allowed_lists)):
                f.write(json.dumps({"idx": i, "text": txt, "gold": g, "pred": p, "allowed": alw}) + "\n")

    del e5_model, e5_tok
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrate_threshold_on_dev_fn(
    dev_probs: np.ndarray,
    dev_gold: List[List[str]],
    label_list: List[str],
    thresh_grid: np.ndarray,
    min_labels: int,
    max_labels: int,
) -> float:
    return calibrate_threshold_on_dev(
        dev_probs=dev_probs,
        dev_gold=dev_gold,
        label_list=label_list,
        thresh_grid=thresh_grid,
        min_labels=min_labels,
        max_labels=max_labels,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=Path, default=Path("train.tsv"))
    ap.add_argument("--dev_path", type=Path, default=Path("dev.tsv"))
    ap.add_argument("--test_path", type=Path, default=Path("test.tsv"))

    ap.add_argument("--title_col", type=str, default="title")
    ap.add_argument("--abstract_col", type=str, default="abstract")
    ap.add_argument("--labels_col", type=str, default="labels")

    ap.add_argument("--output_base_dir", type=Path, default=Path("bert_family_outputs"))
    ap.add_argument("--label_mapping_path", type=Path, default=None)

    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=16)

    ap.add_argument("--calibrate_threshold_on_dev", action="store_true", default=True)
    ap.add_argument("--thresh_min", type=float, default=0.05)
    ap.add_argument("--thresh_max", type=float, default=0.95)
    ap.add_argument("--thresh_steps", type=int, default=19)

    ap.add_argument("--min_labels", type=int, default=1)
    ap.add_argument("--max_labels", type=int, default=7)

    ap.add_argument("--save_predictions", action="store_true", default=True)

    ap.add_argument("--embed_model_path", type=Path, default=Path("e5-base-v2"))
    ap.add_argument("--top_k_cpc", type=int, default=20)
    ap.add_argument("--cpc_emb_batch", type=int, default=64)
    ap.add_argument("--patent_emb_batch", type=int, default=32)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.label_mapping_path is None:
        args.label_mapping_path = args.output_base_dir / "label_mapping.json"

    for p in [args.train_path, args.dev_path, args.test_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")

    if not args.output_base_dir.is_dir():
        raise FileNotFoundError(f"Missing output_base_dir: {args.output_base_dir}")

    if not args.label_mapping_path.is_file():
        raise FileNotFoundError(f"Missing label_mapping_path: {args.label_mapping_path}")

    if not args.embed_model_path.is_dir():
        raise FileNotFoundError(f"Missing embed_model_path: {args.embed_model_path}")

    label_list = load_label_list_from_label_mapping(args.label_mapping_path)

    thresh_grid = np.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)

    models = [
        {"name": "bert", "run_name": "bert_classifier", "base_tokenizer_dir": "bert-base-uncased"},
        {"name": "scibert", "run_name": "scibert_classifier", "base_tokenizer_dir": "scibert_scivocab_uncased"},
        {"name": "patentsberta", "run_name": "patentsberta_classifier", "base_tokenizer_dir": "PatentSBERTa"},
    ]

    cpc_labels: Dict[str, str] = {
    # A01 — Agriculture; Forestry; Animal Husbandry; Hunting; Fishing
    "A01B": "Soil working in agriculture or forestry; parts, details, or accessories of agricultural machines or implements, in general",
    "A01C": "Planting; sowing; fertilising",
    "A01D": "Harvesting; mowing",
    "A01F": "Threshing; baling straw or hay; mowers combined with threshers; straw elevators",
    "A01G": "Horticulture; cultivation of vegetables, flowers, rice, fruit, vines, hops, or seaweed; forestry; watering",
    "A01H": "New plants or processes for obtaining them; plant reproduction by tissue culture techniques",
    "A01J": "Manufacture of dairy products",
    "A01K": "Animal husbandry; care of birds, fishes, insects; fish breeding; rearing or breeding animals, not otherwise provided for",
    "A01L": "Horseshoeing; shoeing of animals",
    "A01M": "Catching, trapping or scaring of animals",
    "A01N": "Preservation of bodies of humans or animals or plants or parts thereof; biocides, e.g. as disinfectants, pesticides, herbicides; pest repellents or attractants; plant growth regulators",
    "A01P": "Biocidal, pest repellant, pest attractant, or plant growth regulatory activity of chemical compounds or preparations",

    # A21 — Baking
    "A21B": "Bakers' ovens; machines or equipment for baking",
    "A21C": "Machines or equipment for making or processing doughs; handling baked articles",
    "A21D": "Treatment of flour or dough for baking; baking processes",

    # A22 — Butchering; Meat treatment
    "A22B": "Slaughtering",
    "A22C": "Processing meat, poultry, or fish",

    # A23 — Foods or Foodstuffs; Treatment Thereof
    "A23B": "Preservation of foods or foodstuffs, in general",
    "A23C": "Dairy products; their preparation or treatment",
    "A23D": "Edible oils or fats; their treatment",
    "A23F": "Coffee; tea; substitutes; their preparation",
    "A23G": "Cocoa; chocolate; confectionery; ice cream",
    "A23J": "Protein compositions for foodstuffs; working-up proteins for foodstuffs",
    "A23K": "Animal feeding-stuffs; methods of making same",
    "A23L": "Foods, foodstuffs, or non-alcoholic beverages; their preparation or treatment",
    "A23N": "Machines for processing harvested fruit, vegetables, or nuts; slicing, peeling, or removing seeds",
    "A23P": "Shaping or working of foodstuffs",
    "A23V": "Use of food ingredients not otherwise provided for",

    # A24 — Tobacco; Smokers' Requisites
    "A24B": "Manufacture of tobacco; cigars; cigarettes",
    "A24C": "Machines for making cigars or cigarettes",
    "A24D": "Cigarette-paper; mouthpieces; cigar or cigarette holders",
    "A24F": "Smokers’ requisites; matches",

    # A41 — Wearing Apparel
    "A41B": "Shirts; underwear; baby linen; handkerchiefs",
    "A41C": "Corsets; brassieres; bustles; bodies",
    "A41D": "Outerwear; protective garments; accessories",
    "A41F": "Garment supporters; suspenders",
    "A41G": "Artificial flowers; wreaths",
    "A41H": "Sewing patterns; marking or cutting devices for clothing",

    # A42 — Headwear
    "A42B": "Hats; head coverings; hat-making",
    "A42C": "Manufacture of hats or head coverings",

    # A43 — Footwear
    "A43B": "Footwear",
    "A43C": "Manufacture of footwear",
    "A43D": "Machines, tools, equipment, or processes for manufacturing footwear",

    # A44 — Haberdashery; Jewellery
    "A44B": "Buttons, pins, buckles, slide fasteners, or the like",
    "A44C": "Jewellery; badges; coin-freed or similar devices",
    "A44D": "Wigs; false hair; hair ornaments",

    # A45 — Hand or Travelling Articles
    "A45B": "Walking sticks; umbrellas; parasols",
    "A45C": "Purses; handbags; travelling bags; wallets",
    "A45D": "Hairdressing or shaving equipment",
    "A45F": "Travelling or camp equipment; supports for garments; personal protection devices",

    # A46 — Brushware
    "A46B": "Brushes; bristles; manufacture thereof",
    "A46D": "Working of bristles for brushes",

    # A47 — Furniture; Domestic Articles or Appliances
    "A47B": "Tables; desks; office furniture",
    "A47C": "Chairs; sofas; beds",
    "A47D": "Furniture specially adapted for children",
    "A47F": "Shop, bar, or display equipment",
    "A47G": "Household or table equipment; cleaning of tableware",
    "A47H": "Clothes hangers; supports for domestic articles",
    "A47J": "Kitchen equipment; coffee mills; spice mills; apparatus for making beverages",
    "A47K": "Sanitary equipment for domestic use; toilet accessories",
    "A47L": "Domestic cleaning; washing or drying machines for linen or clothes",

    # A61 — Medical or Veterinary Science; Hygiene
    "A61B": "Diagnosis; surgery; identification",
    "A61C": "Dentistry; oral or dental hygiene",
    "A61D": "Veterinary instruments; animal treatment",
    "A61F": "Filters implantable into blood vessels; prostheses; orthopaedic devices",
    "A61G": "Transport or accommodation for patients; operating tables or chairs",
    "A61H": "Physical therapy apparatus",
    "A61J": "Containers specially adapted for medical or pharmaceutical purposes",
    "A61K": "Preparations for medical, dental, or toilet purposes",
    "A61L": "Methods or apparatus for sterilising materials",
    "A61M": "Devices for introducing media into, or onto, the body (e.g. syringes, catheters)",
    "A61N": "Electrotherapy; magnetotherapy; radiation therapy; ultrasound therapy",
    "A61P": "Specific therapeutic activity of chemical compounds or medicinal preparations",
    "A61Q": "Specific use of cosmetics or similar toilet preparations",

    # A62 — Life-saving; Fire-fighting
    "A62B": "Devices, apparatus or methods for life-saving",
    "A62C": "Fire-fighting",
    "A62D": "Chemical means for extinguishing fires; processes for making them",

    # A63 — Sports; Games; Amusements
    "A63B": "Apparatus for physical training, gymnastics, swimming, climbing, or fencing; ball games",
    "A63C": "Skates; skis; roller-skates; snowboards",
    "A63D": "Angling; fishing; trapping",
    "A63F": "Card, board, or roulette games; video games; electronic games",
    "A63G": "Merry-go-rounds; swings; sports ground installations",
    "A63H": "Toys; models; puppets",
    "A63J": "Devices for theatre, circus, or amusement",
    "A63K": "Methods or devices for timing, scoring, or indicating results",
    
    # A99 — Miscellaneous
    "A99Z": "Subject matter not otherwise provided for in this section",
    
    # B01–B09: SEPARATING; MIXING
    "B01B": "Boiling; Boiling apparatus; Evaporation; Evaporation apparatus",
    "B01D": "Separation; Filtration; Evaporation; Distillation; Absorption; Adsorption",
    "B01F": "Mixing, dissolving, emulsifying or dispersing",
    "B01J": "Chemical or physical processes, e.g. catalysis or colloid chemistry",
    "B01L": "Chemical or physical laboratory apparatus for general use",
    "B02B": "Preparing grain for milling; refining granular fruit to commercial products",
    "B02C": "Crushing, pulverising or disintegrating; milling grain",
    "B03B": "Separating solid materials using liquids or pneumatic tables or jigs",
    "B03C": "Magnetic or electrostatic separation of solid materials",
    "B03D": "Flotation; Differential sedimentation",
    "B04B": "Centrifuges",
    "B04C": "Apparatus using free vortex flow, e.g. cyclones",
    "B05B": "Spraying apparatus; Atomising apparatus; Nozzles",
    "B05C": "Apparatus for applying fluent materials to surfaces",
    "B05D": "Processes for applying fluent materials to surfaces",
    "B06B": "Generating or transmitting mechanical vibrations of infrasonic, sonic or ultrasonic frequency",
    "B07B": "Separating solids from solids by sieving, screening, sifting or by using gas currents",
    "B07C": "Sorting individual articles or bulk material piece-meal, e.g. by optical/mechanical properties",
    "B08B": "Cleaning in general; Prevention of fouling",
    "B09B": "Disposal of solid waste; Treatment or transformation of refuse",
    "B09C": "Reclamation of contaminated soil",
    
    # B21–B33: SHAPING
    "B21B": "Rolling of metal",
    "B21C": "Manufacture of metal sheets, wire, rods, tubes or like products otherwise than by rolling",
    "B21D": "Working or processing of sheet metal or metal tubes without removing material; Punching",
    "B21F": "Working or processing of metal wire",
    "B21G": "Making needles, pins or nails of metal",
    "B21H": "Making particular metal objects by rolling",
    "B21J": "Forging; Hammering; Pressing metal; Riveting; Forge furnaces",
    "B21K": "Making forged or pressed metal products",
    "B21L": "Making metal chains",
    "B22C": "Foundry moulding",
    "B22D": "Casting of metals; Casting of other substances by similar processes",
    "B22F": "Working metallic powder; Manufacture of articles from metallic powder",
    "B23B": "Turning; Boring",
    "B23C": "Milling",
    "B23D": "Planing; Slotting; Shearing; Broaching; Sawing; Filing; Scraping",
    "B23F": "Making gears or toothed racks",
    "B23G": "Thread cutting; Working of screws, bolt heads, or nuts",
    "B23H": "Working of metal by the action of a high concentration of electric current (electroerosion)",
    "B23K": "Soldering, welding, cladding, cutting by heat; Working by laser beam",
    "B23P": "Metal-working not otherwise provided for; Combined operations; Universal machine tools",
    "B23Q": "Details, components, or accessories for machine tools; Combinations of machine tools",
    "B24B": "Machines or processes for grinding or polishing",
    "B24C": "Abrasive or related blasting with particulate material",
    "B24D": "Tools for grinding, buffing or sharpening",
    "B25B": "Hand tools or bench devices for fastening, connecting, or holding",
    "B25C": "Hand-held nailing or stapling tools",
    "B25D": "Percussive tools",
    "B25F": "Combination or multi-purpose tools; Components of portable power-driven tools",
    "B25G": "Handles for hand implements",
    "B25H": "Workshop equipment, e.g. marking-out work; Storage means for workshops",
    "B25J": "Manipulators; Chambers provided with manipulation devices (industrial robots)",
    "B26B": "Hand-held cutting tools not otherwise provided for",
    "B26D": "Cutting; Perforating; Punching; Severing",
    "B26F": "Perforating, punching, cutting-out, stamping-out or severing by means other than cutting",
    "B27B": "Saws for wood or similar material",
    "B27C": "Planing, drilling, milling or turning machines for wood or similar material",
    "B27D": "Working veneer or plywood",
    "B27F": "Dovetailing, tenoning, slotting machines for wood; Nailing or stapling machines",
    "B27G": "Accessory machines or apparatus for working wood; Tools for working wood",
    "B27H": "Bending wood or similar material; Cooperage; Making wooden wheels",
    "B27J": "Mechanical working of cane, cork, or similar materials",
    "B27K": "Treating or impregnating wood or similar materials",
    "B27L": "Removing bark; Splitting wood; Making veneer or wood shavings",
    "B27M": "Working of wood not provided for in other subclasses; Manufacture of specific articles",
    "B27N": "Manufacture by dry processes of wood-based articles (particle board, fiberboard)",
    "B28B": "Shaping clay or other ceramic compositions; Shaping mixtures with cement",
    "B28C": "Preparing clay; Producing mixtures with clay or cement",
    "B28D": "Working stone or stone-like materials",
    "B29B": "Preparation or pretreatment of material to be shaped; Recycling of plastics",
    "B29C": "Shaping or joining of plastics; Shaping of material in a plastic state; Repairing",
    "B29D": "Producing particular articles from plastics or substances in a plastic state",
    "B29K": "Indexing scheme related to moulding materials or mould components",
    "B29L": "Indexing scheme related to articles produced by moulding",
    "B30B": "Presses in general",
    "B31B": "Making containers of paper, cardboard, or similar material",
    "B31C": "Making wound articles of paper or similar material",
    "B31D": "Making articles of paper not provided for elsewhere",
    "B31F": "Mechanical working or deformation of paper or similar material",
    "B32B": "Layered products, e.g. laminated or honeycomb structures",
    "B33Y": "Additive manufacturing, e.g. 3D printing or stereolithography",
    
    # B41–B44: PRINTING & DECORATION
    "B41B": "Machines or accessories for making or setting type; Type; Composing devices",
    "B41C": "Processes for the manufacture or reproduction of printing surfaces",
    "B41D": "Apparatus for the mechanical reproduction of printing surfaces",
    "B41F": "Printing machines or presses",
    "B41G": "Apparatus for bronze printing, line printing, bordering or edging sheets",
    "B41J": "Typewriters; Selective printing mechanisms (inkjet, laser printing, etc.)",
    "B41K": "Stamps; Stamping or numbering apparatus",
    "B41L": "Apparatus for manifolding, duplicating or printing for office/commercial use",
    "B41M": "Printing, duplicating, marking, or copying processes; Colour printing",
    "B41N": "Printing plates or foils; Materials for printing surfaces",
    "B41P": "Indexing scheme related to printing or stamps",
    "B42B": "Bookbinding; Permanently attaching sheets or signatures",
    "B42C": "Bookbinding processes or apparatus",
    "B42D": "Books; Book covers; Printed matter with identification or security features",
    "B42F": "Sheets temporarily attached together; Filing appliances",
    "B42P": "Indexing scheme for books or filing appliances",
    "B43K": "Implements for writing or drawing",
    "B43L": "Articles for writing upon; Writing or drawing aids",
    "B43M": "Bureau accessories not otherwise provided for",
    "B44B": "Machines, apparatus or tools for artistic work (engraving, carving, etc.)",
    "B44C": "Producing decorative effects; Mosaics; Paperhanging",
    "B44D": "Painting or artistic drawing; Preserving paintings; Artistic surface finishes",
    "B44F": "Special designs or pictures",
    
    # B60–B68: TRANSPORTING
    "B60B": "Vehicle wheels; Axles; Castors; Increasing wheel adhesion",
    "B60C": "Vehicle tyres; Inflation; Tyre changing; Valves for inflatable bodies",
    "B60D": "Vehicle connections (e.g. towing, coupling)",
    "B60F": "Vehicles for use on rail and road; Amphibious vehicles; Convertible vehicles",
    "B60G": "Vehicle suspension arrangements",
    "B60H": "Heating, cooling, or ventilating in vehicles",
    "B60J": "Windows, windscreens, non-fixed roofs, doors, or protective coverings for vehicles",
    "B60K": "Arrangement of propulsion units or transmissions; Auxiliary drives; Dashboards",
    "B60L": "Electric propulsion of vehicles; Power supply for electric vehicles; Electrodynamic brakes",
    "B60M": "Power supply lines for electrically-propelled vehicles (e.g., rail systems)",
    "B60N": "Vehicle seats; Passenger accommodation",
    "B60P": "Vehicles adapted for load transportation or carrying special loads",
    "B60Q": "Arrangement of signalling or lighting devices for vehicles",
    "B60R": "Vehicle fittings or parts not otherwise provided for",
    "B60S": "Servicing, cleaning, repairing, or supporting vehicles",
    "B60T": "Vehicle brake control systems or parts thereof",
    "B60V": "Air-cushion vehicles",
    "B60W": "Conjoint control of vehicle sub-units; Hybrid control; Autonomous driving",
    "B60Y": "Indexing scheme for cross-cutting vehicle technology",
    "B61B": "Railway systems; Ropeways; Cable cars",
    "B61C": "Locomotives; Motor railcars",
    "B61D": "Railway vehicles; Carriages; Wagons",
    "B61F": "Rail vehicle suspensions; Bogies; Wheel guards",
    "B61G": "Couplings; Buffers; Draw gear for rail vehicles",
    "B61H": "Brakes or retarders for rail vehicles",
    "B61J": "Shunting or marshalling of rail vehicles",
    "B61K": "Auxiliary railway equipment",
    "B61L": "Guiding railway traffic; Safety systems for railways",
    "B62B": "Hand-propelled vehicles; Perambulators; Sledges",
    "B62C": "Vehicles drawn by animals",
    "B62D": "Motor vehicles; Trailers; Steering systems",
    "B62H": "Cycle stands; Locks; Anti-theft devices for bicycles",
    "B62J": "Cycle saddles or seats; Cycle accessories",
    "B62K": "Cycles; Frames; Steering; Sidecars",
    "B62L": "Cycle brakes",
    "B62M": "Rider or powered propulsion of cycles; Transmissions for cycles",
    "B63B": "Ships or other waterborne vessels; Equipment for shipping",
    "B63C": "Launching, hauling-out, or dry-docking of vessels; Life-saving in water",
    "B63G": "Offensive or defensive arrangements on vessels; Submarines; Mine-sweeping",
    "B63H": "Marine propulsion or steering",
    "B63J": "Auxiliaries on vessels (ventilation, power generation, etc.)",
    "B64B": "Lighter-than-air aircraft",
    "B64C": "Aeroplanes; Helicopters",
    "B64D": "Equipment for aircraft; Flight suits; Parachutes",
    "B64F": "Ground installations for aircraft; Maintenance or repair facilities",
    "B64G": "Cosmonautics; Space vehicles or equipment therefor",
    "B64U": "Unmanned aerial vehicles (UAVs); Drones; Equipment therefor",
    "B65B": "Packaging machines or methods",
    "B65C": "Labelling or tagging machines",
    "B65D": "Containers for storage or transport; Packaging elements; Packages",
    "B65F": "Gathering or removal of domestic or industrial refuse",
    "B65G": "Transport or storage devices; Conveyors; Pneumatic tubes",
    "B65H": "Handling thin or filamentary materials (sheets, webs, cables)",
    "B66B": "Elevators; Escalators; Moving walkways",
    "B66C": "Cranes; Load-engaging devices",
    "B66D": "Capstans; Winches; Tackles; Hoists",
    "B66F": "Hoisting, lifting, or pushing not otherwise provided for",
    "B67B": "Applying closures to bottles or containers; Opening closed containers",
    "B67C": "Cleaning, filling or emptying bottles or containers",
    "B67D": "Dispensing, delivering, or transferring liquids",
    "B68B": "Harness; Whips",
    "B68C": "Saddles; Stirrups",
    "B68F": "Making articles from leather, canvas or similar material",
    "B68G": "Upholstering; Upholstery not otherwise provided for",
    "B81B": "Microstructural devices or systems, e.g. MEMS",
    "B81C": "Processes or apparatus for making microstructural devices",
    "B82B": "Nanostructures formed by manipulation of atoms or molecules",
    "B82Y": "Specific uses or applications of nanostructures",
    "B99Z": "Subject matter not otherwise provided for in section B",

    # C01–C06: INORGANIC & BASIC CHEMISTRY
    "C01B": "Non-metallic elements; Compounds thereof",
    "C01C": "Ammonia; Cyanogen; Compounds containing carbon–nitrogen bonds",
    "C01D": "Halogen or sulfur compounds of nonmetals",
    "C01F": "Compounds of the alkali or alkaline-earth metals",
    "C01G": "Compounds of metals, not otherwise provided for",
    "C01P": "Indexing scheme for chemical or physical properties of inorganic compounds",
    "C02F": "Treatment of water, wastewater, sewage, or sludge",
    "C03B": "Manufacture of glass or vitreous products",
    "C03C": "Chemical composition of glass; Surface treatment of glass",
    "C04B": "Lime; Cement; Mortar; Concrete; Artificial stone",
    "C05B": "Phosphatic fertilizers",
    "C05C": "Nitrogenous fertilizers",
    "C05D": "Potassic fertilizers",
    "C05F": "Organic fertilizers",
    "C05G": "Mixed fertilizers; Fertilizer mixtures",
    "C06B": "Explosives or thermic compositions; Manufacture thereof; Matches",
    "C06C": "Detonating or priming compositions",
    "C06D": "Igniting compositions; Pyrotechnics",
    "C06F": "Matches; Pyrophoric compositions",

    # C07–C11: ORGANIC & MACROMOLECULAR CHEMISTRY
    "C07B": "General methods of organic chemistry; Apparatus therefor",
    "C07C": "Acyclic or carbocyclic compounds",
    "C07D": "Heterocyclic compounds",
    "C07F": "Acyclic, carbocyclic or heterocyclic compounds containing elements other than carbon, hydrogen, halogen, oxygen, nitrogen",
    "C07G": "Compounds of unknown constitution",
    "C07H": "Sugars; Derivatives thereof; Nucleosides; Nucleotides; Nucleic acids",
    "C07J": "Steroids",
    "C07K": "Peptides",
    "C08B": "Polysaccharides; Derivatives thereof",
    "C08C": "Treatment or chemical modification of macromolecular substances",
    "C08F": "Macromolecular compounds obtained by polymerising unsaturated monomers",
    "C08G": "Macromolecular compounds obtained otherwise than by polymerisation of unsaturated monomers",
    "C08H": "Derivatives of natural macromolecular compounds",
    "C08J": "Working-up; Shaping; Treating compositions of macromolecular substances",
    "C08K": "Use of inorganic or non-macromolecular organic substances as ingredients in compositions",
    "C08L": "Compositions of macromolecular compounds",
    "C09B": "Dyes; Pigments; Bleaching agents",
    "C09C": "Treatment of inorganic pigments or dyestuffs; Preparation thereof",
    "C09D": "Coating compositions; Paints; Varnishes",
    "C09F": "Polishing compositions; Detergent compositions; Soaps",
    "C09G": "Polishes; Adhesives; Mastics; Cements",
    "C09H": "Adhesives; Non-mechanical connections",
    "C09J": "Adhesive compositions; Use of adhesives",
    "C09K": "Materials for specific applications (lubricants, fuels, heat-transfer agents)",
    "C10B": "Destructive distillation of carbonaceous material",
    "C10C": "Working-up tar, pitch or similar products",
    "C10F": "Hydrocarbon oils; Treatment of hydrocarbon oils",
    "C10G": "Cracking of hydrocarbon oils; Reforming; Recovery of by-products",
    "C10H": "Hydrogenation or dehydrogenation of hydrocarbons",
    "C10J": "Production of producer gas, water gas, synthesis gas",
    "C10K": "Purification of gases or vapours",
    "C10L": "Fuels not otherwise provided for; Natural gas; Synthetic fuels",
    "C10M": "Lubricating compositions; Use of chemical substances as lubricants",
    "C10N": "Additives for lubricants",
    "C11B": "Manufacture of animal or vegetable oils, fats, fatty acids or waxes",
    "C11C": "Refining fats, oils or waxes",
    "C11D": "Detergent compositions; Use of substances as cleaning agents",

    # C12–C14: BIOCHEMISTRY, MICROBIOLOGY
    "C12C": "Brewing of beer; Manufacture of malt",
    "C12F": "Fermentation or enzyme-using processes for producing alcohols or beverages",
    "C12G": "Wine; Other alcoholic beverages; Preparation thereof",
    "C12H": "Production of organic compounds or elements using fermentation or enzymes",
    "C12J": "Fermentation or enzyme processes for treatment of dough or baking",
    "C12L": "Fermentation apparatus; Enzyme reactors",
    "C12M": "Apparatus for enzymology or microbiology",
    "C12N": "Microorganisms or enzymes; Compositions thereof; Genetic engineering; Cell culture",
    "C12P": "Fermentation or enzymatic synthesis of chemical compounds",
    "C12Q": "Measuring or testing processes involving enzymes or microorganisms",
    "C12R": "Indexing scheme for microorganisms used in C12N, C12P, C12Q",
    "C12Y": "Enzyme or microorganism classification; Indexing schemes",
    "C13B": "Production of sugar from sugar-containing material",
    "C13K": "Extraction of non-sugar components from molasses or sugar juice",
    "C14B": "Mechanical treatment of hides, skins or leather in preparation for tanning",
    "C14C": "Tanning; Impregnating hides, skins or leather",

    # C21–C25: METALLURGY
    "C21B": "Manufacture of iron or steel",
    "C21C": "Processing of pig-iron; Manufacture of wrought-iron or steel",
    "C21D": "Modification of physical structure of ferrous metals (heat treatment)",
    "C22B": "Production and refining of metals",
    "C22C": "Alloys",
    "C22F": "Changing physical structure of non-ferrous metals or alloys",
    "C23C": "Coating metallic materials; Vacuum evaporation; Sputtering; Ion implantation",
    "C23D": "Electroplating; Electroforming",
    "C23F": "Non-mechanical removal of metallic material (etching, pickling)",
    "C23G": "Cleaning or de-greasing metallic material",
    "C25B": "Electrolytic or electrophoretic processes",
    "C25C": "Electrolytic production of inorganic compounds or non-metals",
    "C25D": "Electroplating; Electrolytic coating processes",
    "C25F": "Electrolytic or electrophoretic removal of material",
    "C30B": "Single-crystal growth; Production of polycrystalline materials",
    "C40B": "Combinatorial chemistry; Libraries of chemical compounds",
    "C99Z": "Subject matter not otherwise provided for in section C",

    # D01 – Fibres and spinning
    "D01B": "Mechanical treatment of natural fibrous or filamentary material to obtain fibres or filaments",
    "D01C": "Chemical or biological treatment of natural fibrous material to obtain filaments or fibres for spinning",
    "D01D": "Mechanical methods or apparatus in the manufacture of artificial filaments, threads, fibres, bristles or ribbons",
    "D01F": "Chemical features in the manufacture of artificial filaments, threads, fibres, bristles or ribbons; apparatus for carbon filaments",
    "D01G": "Preliminary treatment of fibres, e.g. for spinning",
    "D01H": "Spinning or twisting of fibres, filaments, or yarns",

    # D02 – Yarns and ropes
    "D02G": "Crimping or curling fibres, filaments, threads, or yarns; yarns or threads",
    "D02H": "Warping, beaming or leasing",
    "D02J": "Finishing or dressing of filaments, yarns, threads, cords or ropes",

    # D03 – Weaving
    "D03C": "Shedding mechanisms; pattern cards or chains; punching of cards; designing patterns",
    "D03D": "Woven fabrics; methods of weaving; looms",
    "D03J": "Auxiliary weaving apparatus; weavers’ tools; shuttles",

    # D04 – Braiding, lace-making, knitting, non-wovens
    "D04B": "Knitting",
    "D04C": "Braiding or manufacture of lace, including bobbin-net lace; braiding machines",
    "D04D": "Trimmings; ribbons, tapes or bands not otherwise provided for",
    "D04G": "Making nets by knotting; making knotted carpets or tapestries; knotting in general",
    "D04H": "Making textile fabrics other than by weaving, knitting or braiding; non-woven fabrics; felts; cotton-wool; wadding",

    # D05 – Sewing, embroidering, tufting
    "D05B": "Sewing; sewing apparatus or machines; sewing processes",
    "D05C": "Embroidering; tufting",
    "D05D": "Indexing scheme for sewing, embroidering and tufting",

    # D06 – Textile treatment, laundering, dyeing, decorating
    "D06B": "Treating textile materials using liquids, gases or vapours",
    "D06C": "Finishing, dressing, tentering or stretching textile fabrics",
    "D06F": "Laundering, drying, ironing, pressing or folding textile articles",
    "D06G": "Mechanical or pressure cleaning of carpets, rugs, sacks, hides, or other skin or textile articles",
    "D06H": "Marking, inspecting, seaming or severing textile materials",
    "D06J": "Pleating, kilting or goffering textile fabrics or wearing apparel",
    "D06L": "Dry-cleaning, washing or bleaching fibres, threads, yarns, fabrics, feathers or fibrous goods",
    "D06M": "Treatment, not provided for elsewhere, of fibres, threads, yarns, fabrics, feathers or fibrous goods",
    "D06N": "Wall, floor or like covering materials consisting of fibrous webs coated with macromolecular material; flexible sheet materials",
    "D06P": "Dyeing or printing textiles; dyeing leather, furs or solid macromolecular substances",
    "D06Q": "Decorating textiles by local treatment or surface effects",

    # D07 – Ropes and cables
    "D07B": "Ropes or cables in general",

    # D10 – Indexing scheme
    "D10B": "Indexing scheme associated with subclasses of Section D relating to textiles",

    # D21 – Paper and cellulose
    "D21B": "Fibrous raw materials or their mechanical treatment",
    "D21C": "Production of cellulose by removing non-cellulose substances; regeneration of pulping liquors",
    "D21D": "Treatment of the materials before passing to the paper-making machine",
    "D21F": "Paper-making machines; methods of producing paper thereon",
    "D21G": "Calenders; accessories for paper-making machines",
    "D21H": "Pulp compositions; impregnation or coating of paper; treatment of finished paper",
    "D21J": "Fibreboard; manufacture of articles from cellulosic fibrous suspensions or papier-mâché",

    # D99 – Miscellaneous
    "D99Z": "Subject matter not otherwise provided for in section D",

    # E01 – Construction of roads, railways or bridges
    "E01B": "Permanent way; tools and machines for making railways of all kinds",
    "E01C": "Construction of or surfaces for roads, sports grounds or similar; machines or tools for road construction or repair",
    "E01D": "Construction or assembly of bridges, elevated roadways or viaducts",
    "E01F": "Additional work in road construction such as equipment, platforms, helicopter landing stages, signs, snow fences",
    "E01H": "Street or permanent-way cleaning; cleaning beaches; dispersing or preventing fog in general",

    # E02 – Hydraulic engineering; foundations; soil shifting
    "E02B": "Hydraulic engineering; waterways, dams, or hydraulic structures",
    "E02C": "Ship-lifting devices or mechanisms",
    "E02D": "Foundations; excavations; embankments; underground or underwater structures",
    "E02F": "Dredging; soil-shifting; excavating or loading equipment",

    # E03 – Water supply; sewerage
    "E03B": "Installations or methods for obtaining, collecting or distributing water",
    "E03C": "Domestic plumbing installations for fresh or waste water; sinks",
    "E03D": "Water-closets or urinals with flushing devices; flushing valves",
    "E03F": "Sewers; cesspools; storm-water drainage",

    # E04 – Building
    "E04B": "General building constructions; walls; roofs; floors; ceilings; insulation or protection of buildings",
    "E04C": "Structural elements; building materials",
    "E04D": "Roof coverings; sky-lights; gutters; roof-working tools",
    "E04F": "Finishing work on buildings, e.g. stairs, floors",
    "E04G": "Scaffolding; formwork; shuttering; building implements or aids; repairing or breaking up existing buildings",
    "E04H": "Buildings or like structures for particular purposes; swimming pools; masts; fencing; tents or canopies",

    # E05 – Locks, keys, fittings, safes
    "E05B": "Locks; accessories therefor; handcuffs",
    "E05C": "Bolts or fastening devices for doors or windows",
    "E05D": "Hinges or suspension devices for doors, windows or wings",
    "E05F": "Devices for moving wings into open or closed position; checks for wings; wing fittings",
    "E05G": "Safes or strong-rooms; bank protection devices; safety transaction partitions",
    "E05Y": "Indexing scheme associated with E05D and E05F (construction elements, electric control, user interfaces, etc.)",

    # E06 – Doors, windows, ladders
    "E06B": "Fixed or movable closures for openings in buildings, vehicles or fences, e.g. doors, windows, blinds, gates",
    "E06C": "Ladders; ladder accessories",

    # E21 – Earth drilling; mining
    "E21B": "Earth or rock drilling; obtaining oil, gas, water, soluble or meltable materials from wells",
    "E21C": "Mining or quarrying methods or apparatus",
    "E21D": "Shafts; tunnels; galleries; large underground chambers",
    "E21F": "Safety devices, transport, filling-up, rescue, ventilation or draining in or of mines or tunnels",

    # E99 – Miscellaneous
    "E99Z": "Subject matter not otherwise provided for in section E",

    # F01 – Machines or engines in general
    "F01B": "Machines or engines in general or of positive-displacement type, e.g. steam engines",
    "F01C": "Rotary-piston or oscillating-piston machines or engines",
    "F01D": "Non-positive displacement machines or engines, e.g. turbines",
    "F01K": "Steam engine plants; engine plants not otherwise provided for; engines using special working fluids or cycles",
    "F01L": "Cyclically operating valves for machines or engines",
    "F01M": "Lubricating of machines or engines; crankcase ventilating",
    "F01N": "Gas-flow silencers or exhaust apparatus for machines or engines",
    "F01P": "Cooling of machines or engines in general",

    # F02 – Combustion engines; jet propulsion
    "F02B": "Internal-combustion piston engines; combustion engines in general",
    "F02C": "Gas-turbine plants; air intakes for jet-propulsion plants; controlling fuel supply in air-breathing jet plants",
    "F02D": "Controlling combustion engines",
    "F02F": "Cylinders, pistons or casings for combustion engines; arrangements of sealings",
    "F02G": "Hot-gas or combustion-product positive-displacement engine plants; use of waste heat of combustion engines",
    "F02K": "Jet-propulsion plants",
    "F02M": "Supplying combustion engines with combustible mixtures or constituents thereof",
    "F02N": "Starting of combustion engines; starting aids",
    "F02P": "Ignition for internal-combustion engines; testing of ignition timing",

    # F03 – Machines or engines for liquids; wind or spring motors
    "F03B": "Machines or engines for liquids (e.g. water turbines, hydraulic motors)",
    "F03C": "Positive-displacement engines driven by liquids",
    "F03D": "Wind motors; wind turbines",
    "F03G": "Spring, weight, inertia or like motors; mechanical power from energy sources not otherwise provided for",
    "F03H": "Producing reactive propulsive thrust not otherwise provided for",

    # F04 – Pumps for liquids or elastic fluids
    "F04B": "Positive-displacement machines for liquids; pumps",
    "F04C": "Rotary-piston or oscillating-piston positive-displacement pumps or machines",
    "F04D": "Non-positive-displacement pumps",
    "F04F": "Pumping of fluid by direct contact of another fluid or by using fluid inertia; siphons",

    # F05 – Indexing schemes for engines or pumps
    "F05B": "Indexing scheme relating to wind, spring, weight, inertia motors or engines for liquids (internal scheme)",
    "F05C": "Indexing scheme relating to materials or properties of machines, engines or pumps (internal scheme)",
    "F05D": "Indexing scheme for aspects relating to non-positive displacement machines or jet-propulsion plants",

    # F15 – Fluid-pressure systems; hydraulics; pneumatics
    "F15B": "Systems acting by means of fluids in general; fluid-pressure actuators; servomotors; details of systems",
    "F15C": "Fluid-circuit elements predominantly used for computing or control purposes",
    "F15D": "Fluid dynamics; methods or means for influencing flow of gases or liquids",

    # F16 – Engineering elements and units
    "F16B": "Devices for fastening or securing constructional elements or machine parts together; joints or jointing",
    "F16C": "Shafts; bearings; flexible shafts; elements or crankshaft mechanisms; rotary bodies other than gearing elements",
    "F16D": "Couplings for transmitting rotation; clutches; brakes",
    "F16F": "Springs; shock-absorbers; means for damping vibration",
    "F16G": "Belts, cables, or ropes; chains; fittings therefor",
    "F16H": "Gearing",
    "F16J": "Pistons; cylinders; sealings",
    "F16K": "Valves; taps; cocks; actuating-floats; venting or aerating devices",
    "F16L": "Pipes; joints or fittings for pipes; supports for pipes or cables; thermal insulation in general",
    "F16M": "Frames, casings or beds of engines, machines or apparatus; stands; supports",
    "F16N": "Lubricating systems",
    "F16P": "Safety devices in general",
    "F16S": "Constructional elements in general; structures built-up from such elements",
    "F16T": "Steam traps or like apparatus for draining-off liquids from enclosures containing gases or vapours",

    # F17 – Storing or distributing gases or liquids
    "F17B": "Gas-holders of variable capacity",
    "F17C": "Vessels for containing or storing compressed, liquefied, or solidified gases",
    "F17D": "Pipe-line systems; pipe-lines",

    # F21 – Lighting
    "F21H": "Incandescent mantles; other incandescent bodies heated by combustion",
    "F21K": "Non-electric light sources; light sources using combustion, luminescence, or semiconductors",
    "F21L": "Portable lighting devices or systems; specially adapted for transportation",
    "F21S": "Non-portable lighting devices; vehicle exterior lighting systems",
    "F21V": "Functional features or details of lighting devices; combinations with other articles",
    "F21W": "Indexing scheme relating to uses or applications of lighting devices or systems",
    "F21Y": "Indexing scheme relating to the form or kind of light sources or colour of emitted light",

    # F22 – Steam generation
    "F22B": "Methods of steam generation; steam boilers",
    "F22D": "Preheating or accumulating feed-water for steam generation; water level control; auxiliary devices",
    "F22G": "Superheating of steam",

    # F23 – Combustion apparatus; combustion processes
    "F23B": "Combustion using only solid fuel",
    "F23C": "Combustion using fluid fuel or solid fuel suspended in air",
    "F23D": "Burners",
    "F23G": "Cremation furnaces; consuming waste products by combustion",
    "F23H": "Grates; cleaning or raking grates",
    "F23J": "Removal or treatment of combustion products or residues; flues",
    "F23K": "Feeding fuel to combustion apparatus",
    "F23L": "Supplying air or non-combustible gases to combustion apparatus; inducing draught; chimney tops",
    "F23M": "Casings, linings or walls for combustion chambers; devices for deflecting air or flames",
    "F23N": "Regulating or controlling combustion",
    "F23Q": "Ignition or extinguishing devices",
    "F23R": "Generating combustion products of high pressure or high velocity (e.g. gas-turbine combustion chambers)",

    # F24 – Heating; ranges; ventilation
    "F24B": "Domestic stoves or ranges for solid fuels",
    "F24C": "Domestic stoves or ranges for liquid, gaseous or other fuels; details of stoves or ranges",
    "F24D": "Domestic or space-heating systems; hot-water supply systems",
    "F24F": "Air-conditioning; air-humidification; ventilation; use of air currents for screening",
    "F24H": "Fluid heaters, e.g. water or air heaters; heat pumps",
    "F24S": "Solar heat collectors; solar heat systems",
    "F24T": "Geothermal collectors; geothermal systems",
    "F24V": "Collection, production or use of heat not otherwise provided for",

    # F25 – Refrigeration or cooling; heat pumps
    "F25B": "Refrigeration machines, plants or systems; heat pump systems",
    "F25C": "Producing, working or handling ice",
    "F25D": "Refrigerators; cold rooms; ice-boxes; cooling or freezing apparatus not otherwise provided for",
    "F25J": "Liquefaction, solidification or separation of gases by pressure and cold treatment",

    # F26 – Drying
    "F26B": "Drying solid materials or objects by removing liquid therefrom",

    # F27 – Furnaces, kilns, ovens, retorts
    "F27B": "Furnaces, kilns, ovens or retorts in general; open sintering apparatus",
    "F27D": "Details or accessories of furnaces, kilns, ovens or retorts",
    "F27M": "Indexing scheme relating to aspects of charges or furnaces, kilns or ovens",

    # F28 – Heat exchange in general
    "F28B": "Steam or vapour condensers",
    "F28C": "Heat-exchange apparatus with direct contact of media without chemical interaction",
    "F28D": "Heat-exchange apparatus without direct contact between media",
    "F28F": "Details of heat-exchange and heat-transfer apparatus",
    "F28G": "Cleaning of internal or external surfaces of heat-exchange or heat-transfer conduits",

    # F41 – Weapons
    "F41A": "Functional features or details common to both smallarms and ordnance; mountings",
    "F41B": "Weapons for projecting missiles without explosive or combustible propellant charge",
    "F41C": "Smallarms, e.g. pistols, rifles; accessories therefor",
    "F41F": "Apparatus for launching projectiles or missiles from barrels; rocket or torpedo launchers",
    "F41G": "Weapon sights; aiming devices",
    "F41H": "Armour; armoured vehicles; camouflage or defence means",
    "F41J": "Targets; target ranges; bullet catchers",

    # F42 – Ammunition; blasting
    "F42B": "Explosive charges, ammunition, fireworks",
    "F42C": "Ammunition fuzes; arming or safety means therefor",
    "F42D": "Blasting",

    # F99 – Miscellaneous
    "F99Z": "Subject matter not otherwise provided for in section F",

    # G01 – Measuring, testing
    "G01B": "Measuring length, thickness or similar linear dimensions; measuring angles; measuring irregularities of surfaces or contours",
    "G01C": "Measuring distances, levels or bearings; surveying; navigation; gyroscopic instruments; photogrammetry or videogrammetry",
    "G01D": "Measuring not specially adapted for a specific variable; arrangements for measuring multiple variables; tariff metering apparatus",
    "G01F": "Measuring volume, volume flow, mass flow or liquid level; metering by volume",
    "G01G": "Weighing",
    "G01H": "Measurement of mechanical vibrations or ultrasonic, sonic or infrasonic waves",
    "G01J": "Measurement of optical radiation characteristics; colorimetry; radiation pyrometry",
    "G01K": "Measuring temperature; measuring quantity of heat; thermally-sensitive elements not otherwise provided for",
    "G01L": "Measuring force, stress, torque, work, mechanical power, mechanical efficiency, or fluid pressure",
    "G01M": "Testing static or dynamic balance of machines or structures; testing of structures or apparatus, not otherwise provided for",
    "G01N": "Investigating or analysing materials by determining their chemical or physical properties",
    "G01P": "Measuring linear or angular speed, acceleration, deceleration, or shock; indicating movement or direction",
    "G01Q": "Scanning-probe techniques or apparatus; applications thereof, e.g. scanning probe microscopy",
    "G01R": "Measuring electric or magnetic variables; testing electrical or magnetic devices",
    "G01S": "Radio direction-finding; radio navigation; determining distance or velocity by use of radio or analogous waves",
    "G01T": "Measurement of nuclear or X-radiation",
    "G01V": "Geophysics; gravitational measurements; detecting masses or objects; tags",
    "G01W": "Meteorology",

    # G02 – Optics
    "G02B": "Optical elements, systems or apparatus",
    "G02C": "Spectacles; sunglasses or goggles; contact lenses",
    "G02F": "Optical devices or arrangements for the control of light by modification of optical properties; non-linear optics; optical logic elements",

    # G03 – Photography and related
    "G03B": "Apparatus or arrangements for taking, projecting or viewing photographs; analogous techniques using other waves",
    "G03C": "Photosensitive materials for photographic purposes; photographic processes",
    "G03D": "Apparatus for processing exposed photographic materials; accessories therefor",
    "G03F": "Photomechanical production of textured or patterned surfaces; semiconductor device processing",
    "G03G": "Electrography; electrophotography; magnetography",
    "G03H": "Holographic processes or apparatus",

    # G04 – Horology
    "G04B": "Mechanically-driven clocks or watches; mechanical parts of clocks or watches",
    "G04C": "Electromechanical clocks or watches",
    "G04D": "Apparatus or tools for making or maintaining clocks or watches",
    "G04F": "Time-interval measuring; apparatus for producing or measuring time intervals",
    "G04G": "Electronic time-pieces",
    "G04R": "Radio-controlled time-pieces",

    # G05 – Control and regulation
    "G05B": "Control or regulating systems in general; monitoring or testing arrangements for such systems",
    "G05D": "Systems for controlling or regulating non-electric variables",
    "G05F": "Systems for regulating electric or magnetic variables",
    "G05G": "Control devices or systems characterised by mechanical features only",

    # G06 – Computing, calculating, counting
    "G06C": "Digital computers in which computation is effected mechanically",
    "G06D": "Digital fluid-pressure computing devices",
    "G06E": "Optical computing devices",
    "G06F": "Electric digital data processing",
    "G06G": "Analogue computers",
    "G06J": "Hybrid computing arrangements (part digital, part analogue)",
    "G06K": "Graphical data reading; presentation of data; record carriers; handling record carriers",
    "G06M": "Counting mechanisms; counting of objects not otherwise provided for",
    "G06N": "Computing arrangements based on specific computational models",
    "G06Q": "ICT specially adapted for administrative, commercial, financial, managerial or supervisory purposes",
    "G06T": "Image data processing or generation",
    "G06V": "Image or video recognition or understanding",

    # G07 – Checking devices
    "G07B": "Ticket-issuing, fare-registering or franking apparatus",
    "G07C": "Time or attendance registers; registering or indicating working of machines; generating random numbers; voting or lottery apparatus",
    "G07D": "Handling of coins or valuable papers; testing, sorting, counting, dispensing, depositing",
    "G07F": "Coin-freed or payment-activated apparatus",
    "G07G": "Registering the receipt of cash, valuables, or tokens",

    # G08 – Signalling
    "G08B": "Signalling or calling systems; order telegraphs; alarm systems",
    "G08C": "Transmission systems for measured values, control or similar signals",
    "G08G": "Traffic control systems",

    # G09 – Education, display, advertising, seals
    "G09B": "Educational or demonstration appliances; teaching aids; models; simulators",
    "G09C": "Ciphering or deciphering apparatus for cryptographic or secrecy purposes",
    "G09D": "Railway or like time or fare tables; perpetual calendars",
    "G09F": "Displaying; advertising; signs; labels or name-plates; seals",
    "G09G": "Arrangements or circuits for control of indicating devices using static means",

    # G10 – Musical instruments; acoustics
    "G10B": "Organs, harmoniums or similar wind musical instruments",
    "G10C": "Pianos, harpsichords, spinets or similar stringed musical instruments with keyboards",
    "G10D": "Stringed, wind, percussion or other musical instruments not otherwise provided for",
    "G10F": "Automatic musical instruments",
    "G10G": "Representation or recording of music; accessories for music or musical instruments",
    "G10H": "Electrophonic musical instruments; instruments generating tones by electromechanical or electronic means",
    "G10K": "Sound-producing devices; acoustic protection; damping noise",
    "G10L": "Speech analysis, synthesis, recognition, processing, or coding",

    # G11 – Information storage
    "G11B": "Information storage based on relative movement between record carrier and transducer",
    "G11C": "Static stores; semiconductor memory devices",

    # G12 – Instrument details
    "G12B": "Constructional details of instruments or comparable details of other apparatus",

    # G16 – ICT for specific applications
    "G16B": "Bioinformatics; genetic or protein-related data processing in computational biology",
    "G16C": "Computational chemistry; chemoinformatics; computational materials science",
    "G16H": "Healthcare informatics; ICT for handling or processing medical or healthcare data",
    "G16Y": "ICT specially adapted for the Internet of Things (IoT)",
    "G16Z": "ICT specially adapted for specific applications, not otherwise provided for",

    # G21 – Nuclear physics; nuclear engineering
    "G21B": "Fusion reactors",
    "G21C": "Nuclear reactors",
    "G21D": "Nuclear power plant",
    "G21F": "Protection against radiation; decontamination arrangements",
    "G21G": "Conversion of chemical elements; radioactive sources",
    "G21H": "Obtaining energy from radioactive sources; applications of radiation not otherwise provided for",
    "G21J": "Nuclear explosives; applications thereof",
    "G21K": "Techniques for handling particles or ionising radiation; irradiation devices",

    # Miscellaneous
    "G99Z": "Subject matter not otherwise provided for in section G",

    # H01 – Basic electric elements
    "H01B": "Cables; conductors; insulators; selection of materials for conductive, insulating or dielectric properties",
    "H01C": "Resistors",
    "H01F": "Magnets; inductances; transformers; selection of materials for magnetic properties",
    "H01G": "Capacitors; electrolytic devices",
    "H01H": "Electric switches; relays; selectors; emergency protective devices",
    "H01J": "Electric discharge tubes or discharge lamps",
    "H01K": "Electric incandescent lamps",
    "H01L": "Semiconductor devices not covered by class H10",
    "H01M": "Processes or means, e.g. batteries, for direct conversion of chemical into electrical energy",
    "H01P": "Waveguides; resonators; lines or other devices of the waveguide type",
    "H01Q": "Antennas, i.e. radio aerials",
    "H01R": "Electrically-conductive connections; coupling devices; current collectors",
    "H01S": "Devices using the process of light amplification by stimulated emission of radiation (lasers)",
    "H01T": "Spark gaps; overvoltage arresters; sparking plugs; corona devices; ion generators",

    # H02 – Generation, conversion or distribution of electric power
    "H02B": "Boards, substations or switching arrangements for the supply or distribution of electric power",
    "H02G": "Installation of electric cables or lines, or of combined optical and electric cables",
    "H02H": "Emergency protective circuit arrangements",
    "H02J": "Circuit arrangements or systems for supplying or distributing electric power; energy storage systems",
    "H02K": "Dynamo-electric machines",
    "H02M": "Apparatus for conversion between AC and DC, or between DC and DC; conversion control or regulation",
    "H02N": "Electric machines not otherwise provided for",
    "H02P": "Control or regulation of electric motors, generators or converters; controlling transformers or choke coils",
    "H02S": "Generation of electric power by conversion of light (e.g. photovoltaic systems)",

    # H03 – Electronic circuitry
    "H03B": "Generation of oscillations by circuits employing active elements in non-switching manner",
    "H03C": "Modulation",
    "H03D": "Demodulation or transference of modulation from one carrier to another",
    "H03F": "Amplifiers",
    "H03G": "Control of amplification",
    "H03H": "Impedance networks; resonant circuits; resonators",
    "H03J": "Tuning resonant circuits; selecting resonant circuits",
    "H03K": "Pulse technique; logic circuits; electronic switching",
    "H03L": "Automatic control, starting, synchronisation or stabilisation of generators of oscillations or pulses",
    "H03M": "Coding; decoding; code conversion in general",

    # H04 – Electric communication technique
    "H04B": "Transmission of information-carrying signals",
    "H04H": "Broadcast communication",
    "H04J": "Multiplex communication",
    "H04K": "Secret communication; jamming of communication",
    "H04L": "Transmission of digital information, e.g. telegraphic communication",
    "H04M": "Telephonic communication",
    "H04N": "Pictorial communication, e.g. television",
    "H04Q": "Selecting; exchange systems; switching networks",
    "H04R": "Loudspeakers, microphones, gramophone pick-ups or like acoustic electromechanical transducers",
    "H04S": "Stereophonic or quadraphonic systems",
    "H04T": "Indexing scheme relating to standards for electric communication technique",
    "H04W": "Wireless communication networks",

    # H05 – Electric techniques not otherwise provided for
    "H05B": "Electric heating; electric light sources not otherwise provided for; circuit arrangements for light sources",
    "H05C": "Electric circuits or apparatus for use in equipment for killing, stunning or guiding living beings",
    "H05F": "Static electricity; naturally-occurring electricity",
    "H05G": "X-ray technique",
    "H05H": "Plasma technique; production or acceleration of charged particles or neutral beams",
    "H05K": "Printed circuits; casings or constructional details of electric apparatus; manufacture of electrical assemblies",

    # H10 – Semiconductor and solid-state devices
    "H10B": "Electronic memory devices",
    "H10D": "Inorganic electric semiconductor devices",
    "H10F": "Inorganic semiconductor devices sensitive to infrared, light or radiation",
    "H10H": "Inorganic light-emitting semiconductor devices having potential barriers",
    "H10K": "Organic electric solid-state devices",
    "H10N": "Electric solid-state devices not otherwise provided for",

    # H99 – Miscellaneous
    "H99Z": "Subject matter not otherwise provided for in section H",

    # Y02 – Climate change mitigation and adaptation technologies
    "Y02A": "Technologies for adaptation to climate change",
    "Y02B": "Climate change mitigation technologies related to buildings (e.g. housing, appliances, end-user applications)",
    "Y02C": "Capture, storage, sequestration or disposal of greenhouse gases (GHG)",
    "Y02D": "Climate change mitigation technologies in information and communication technologies (ICT)",
    "Y02E": "Reduction of greenhouse gas emissions related to energy generation, transmission or distribution",
    "Y02P": "Climate change mitigation technologies in the production or processing of goods",
    "Y02T": "Climate change mitigation technologies related to transportation",
    "Y02W": "Climate change mitigation technologies related to wastewater treatment or waste management",

    # Y04 – ICT impacting other technology areas
    "Y04S": "Systems integrating technologies related to power network operation, communication or information technologies for improving electrical power generation, transmission, distribution or management (smart grids)",

    # Y10 – Former USPC cross-reference collections
    "Y10S": "Technical subjects covered by former USPC cross-reference art collections (XRACs) and digests",
    "Y10T": "Technical subjects covered by former US classification",

    # Y99 – Generic tagging context
    "Y99Z": "Subject matter not otherwise provided for in section Y (general tagging or cross-sectional technologies)"
	}



    set_all_seeds(args.seed)

    for m in models:
        run_dir = args.output_base_dir / m["run_name"]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run dir: {run_dir}")

        evaluate_one_model(
            model_cfg=m,
            label_list=label_list,
            seed=args.seed,
            output_base_dir=args.output_base_dir,
            dev_path=args.dev_path,
            test_path=args.test_path,
            title_col=args.title_col,
            abstract_col=args.abstract_col,
            labels_col=args.labels_col,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
            calibrate_threshold_on_dev=args.calibrate_threshold_on_dev,
            thresh_grid=thresh_grid,
            min_labels=args.min_labels,
            max_labels=args.max_labels,
            save_predictions=args.save_predictions,
            embed_model_path=args.embed_model_path,
            top_k_cpc=args.top_k_cpc,
            cpc_emb_batch=args.cpc_emb_batch,
            patent_emb_batch=args.patent_emb_batch,
            cpc_labels=cpc_labels,
        )


if __name__ == "__main__":
    main()
