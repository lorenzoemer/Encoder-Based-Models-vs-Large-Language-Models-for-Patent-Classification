
from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover
    EmissionsTracker = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    train_path: str
    dev_path: str
    test_path: str
    label_mapping: str
    model_path: str
    output_dir: str

    model_name: str = "encoder"
    input_format: str = "auto"  # auto, tsv, csv, json, jsonl
    title_column: str = "title"
    abstract_column: str = "abstract"
    labels_column: str = "labels"

    offline: bool = False
    overwrite: bool = False
    seed: int = 42

    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum: int = 2
    max_epochs: int = 5
    warmup_ratio: float = 0.06
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    patience: int = 2

    loss: str = "bce"  # bce, focal, cb_bce
    gamma: float = 2.0
    alpha: Optional[float] = None
    cb_beta: float = 0.999
    oversample: bool = False

    min_labels: int = 1
    max_labels: int = 7
    threshold_grid: str = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"
    rare_threshold_freq: int = 50

    use_codecarbon: bool = True
    codecarbon_project_name: str = "encoder_longtail_cpc"
    codecarbon_measure_power_secs: int = 10

    num_workers: int = 2
    pin_memory: bool = True
    max_train_examples: Optional[int] = None
    max_dev_examples: Optional[int] = None
    max_test_examples: Optional[int] = None

    save_all_epoch_checkpoints: bool = False
    eval_label_space_from_mapping_only: bool = True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a BERT-family CPC classifier with long-tail mitigation.")

    p.add_argument("--train-path", required=True)
    p.add_argument("--dev-path", required=True)
    p.add_argument("--test-path", required=True)
    p.add_argument("--label-mapping", required=True)
    p.add_argument("--model-path", required=True, help="Local path or Hugging Face model id.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--model-name", default="encoder")
    p.add_argument("--input-format", default="auto", choices=["auto", "tsv", "csv", "json", "jsonl"])
    p.add_argument("--title-column", default="title")
    p.add_argument("--abstract-column", default="abstract")
    p.add_argument("--labels-column", default="labels")

    p.add_argument("--offline", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)

    p.add_argument("--loss", default="bce", choices=["bce", "focal", "cb_bce"])
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--cb-beta", type=float, default=0.999)
    p.add_argument("--oversample", action="store_true")

    p.add_argument("--min-labels", type=int, default=1)
    p.add_argument("--max-labels", type=int, default=7)
    p.add_argument("--threshold-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    p.add_argument("--rare-threshold-freq", type=int, default=50)

    p.add_argument("--no-codecarbon", action="store_true")
    p.add_argument("--codecarbon-project-name", default="encoder_longtail_cpc")
    p.add_argument("--codecarbon-measure-power-secs", type=int, default=10)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-dev-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)

    p.add_argument("--save-all-epoch-checkpoints", action="store_true")
    p.add_argument("--eval-label-space-include-test-gold", action="store_true")

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        label_mapping=args.label_mapping,
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        input_format=args.input_format,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        labels_column=args.labels_column,
        offline=args.offline,
        overwrite=args.overwrite,
        seed=args.seed,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        max_epochs=args.max_epochs,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        loss=args.loss,
        gamma=args.gamma,
        alpha=args.alpha,
        cb_beta=args.cb_beta,
        oversample=args.oversample,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        threshold_grid=args.threshold_grid,
        rare_threshold_freq=args.rare_threshold_freq,
        use_codecarbon=not args.no_codecarbon,
        codecarbon_project_name=args.codecarbon_project_name,
        codecarbon_measure_power_secs=args.codecarbon_measure_power_secs,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        max_train_examples=args.max_train_examples,
        max_dev_examples=args.max_dev_examples,
        max_test_examples=args.max_test_examples,
        save_all_epoch_checkpoints=args.save_all_epoch_checkpoints,
        eval_label_space_from_mapping_only=not args.eval_label_space_include_test_gold,
    )


# =============================================================================
# Utilities
# =============================================================================

def configure_environment(cfg: Config) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    logging.set_verbosity_info()


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


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def environment_report(cfg: Config) -> Dict[str, Any]:
    try:
        import transformers
    except Exception:
        transformers = None
    try:
        import sklearn
    except Exception:
        sklearn = None
    try:
        import datasets
    except Exception:
        datasets = None
    try:
        import codecarbon
    except Exception:
        codecarbon = None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": getattr(transformers, "__version__", None),
        "sklearn_version": getattr(sklearn, "__version__", None),
        "datasets_version": getattr(datasets, "__version__", None),
        "codecarbon_version": getattr(codecarbon, "__version__", None),
        "config": asdict(cfg),
    }


def parse_threshold_grid(s: str) -> List[float]:
    values = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not values:
        raise ValueError("Threshold grid is empty.")
    return values


# =============================================================================
# Label and data loading
# =============================================================================

def normalize_cpc_label(code: Any) -> str:
    c = str(code or "").strip().upper()
    if not c:
        return ""
    c = c.split("/")[0]
    if len(c) >= 4 and c[0].isalpha() and c[1:3].isdigit():
        return c[:4]
    return c[:4]


def dedupe_preserve_order(labels: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in labels:
        c = normalize_cpc_label(x)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_labels(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return dedupe_preserve_order(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, list):
                    return dedupe_preserve_order(obj)
            except Exception:
                pass
        if ";" in s:
            return dedupe_preserve_order([x.strip() for x in s.split(";")])
        if "," in s:
            return dedupe_preserve_order([x.strip() for x in s.split(",")])
        return dedupe_preserve_order([s])
    return dedupe_preserve_order([raw])


def load_label_list(mapping_path: str) -> List[str]:
    p = Path(mapping_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing label mapping: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        labels = obj["labels"]
    elif isinstance(obj, dict) and "id2label" in obj:
        id2label = obj["id2label"]
        labels = [id2label[str(i)] for i in sorted(map(int, id2label.keys()))]
    elif isinstance(obj, dict) and "label2id" in obj:
        labels = [lab for lab, _ in sorted(obj["label2id"].items(), key=lambda kv: int(kv[1]))]
    elif isinstance(obj, dict) and all(str(k).isdigit() for k in obj.keys()):
        labels = [obj[str(i)] for i in sorted(map(int, obj.keys()))]
    elif isinstance(obj, list):
        labels = obj
    else:
        raise ValueError(f"Unrecognized label mapping format: {p}")

    labels = dedupe_preserve_order(labels)
    if not labels:
        raise ValueError(f"No valid labels parsed from {p}")
    return labels


def infer_dataset_format(path: str, requested: str) -> Tuple[str, Optional[str]]:
    if requested == "tsv":
        return "csv", "\t"
    if requested == "csv":
        return "csv", ","
    if requested in {"json", "jsonl"}:
        return "json", None
    suffix = Path(path).suffix.lower()
    if suffix == ".tsv":
        return "csv", "\t"
    if suffix == ".csv":
        return "csv", ","
    if suffix in {".json", ".jsonl"}:
        return "json", None
    raise ValueError(f"Cannot infer format from {path}. Pass --input-format.")


def build_text(title: Any, abstract: Any) -> str:
    t = str(title or "").strip()
    a = str(abstract or "").strip()
    return f"{t}. {a}" if t and a else (t or a)


def load_split(path: str, cfg: Config, label_set: Set[str], limit: Optional[int]) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    dataset_type, delimiter = infer_dataset_format(path, cfg.input_format)
    kwargs: Dict[str, Any] = {"data_files": {"d": path}}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    ds = load_dataset(dataset_type, **kwargs)["d"]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    required = [cfg.title_column, cfg.abstract_column, cfg.labels_column]
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Available: {ds.column_names}")

    texts, labels, titles, abstracts = [], [], [], []
    for ex in ds:
        title = ex.get(cfg.title_column, "") or ""
        abstract = ex.get(cfg.abstract_column, "") or ""
        labs = [lab for lab in parse_labels(ex.get(cfg.labels_column)) if lab in label_set]
        titles.append(str(title))
        abstracts.append(str(abstract))
        texts.append(build_text(title, abstract))
        labels.append(labs)
    return texts, labels, titles, abstracts


# =============================================================================
# Dataset and sampling
# =============================================================================

class PatentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], label_list: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.label2id = {lab: i for i, lab in enumerate(label_list)}
        self.num_labels = len(label_list)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for lab in self.labels[idx]:
            if lab in self.label2id:
                y[self.label2id[lab]] = 1.0
        item = {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "labels": y}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def compute_label_frequencies(labels: List[List[str]], label_list: List[str]) -> np.ndarray:
    freq = np.zeros(len(label_list), dtype=np.float64)
    lab2id = {lab: i for i, lab in enumerate(label_list)}
    for labs in labels:
        for lab in labs:
            if lab in lab2id:
                freq[lab2id[lab]] += 1.0
    return freq


def compute_sample_weights(labels: List[List[str]], label_list: List[str], label_freq: np.ndarray) -> np.ndarray:
    lab2id = {lab: i for i, lab in enumerate(label_list)}
    max_freq = max(float(label_freq.max()), 1.0)
    inv_freq = np.where(label_freq > 0, max_freq / np.clip(label_freq, 1, None), 1.0)
    weights = np.ones(len(labels), dtype=np.float64)
    for i, labs in enumerate(labels):
        idxs = [lab2id[lab] for lab in labs if lab in lab2id]
        if idxs:
            weights[i] = float(np.mean(inv_freq[idxs]))
    return weights


def make_train_loader(dataset: PatentDataset, labels: List[List[str]], label_list: List[str], label_freq: np.ndarray, cfg: Config) -> DataLoader:
    if cfg.oversample:
        weights = compute_sample_weights(labels, label_list, label_freq)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        return DataLoader(dataset, batch_size=cfg.train_batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, generator=generator, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)


def make_eval_loader(dataset: PatentDataset, cfg: Config) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)


# =============================================================================
# Losses
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            weight = alpha_t * weight
        return (weight * bce).mean()


class ClassBalancedBCE(nn.Module):
    def __init__(self, label_freq: np.ndarray, beta: float = 0.999):
        super().__init__()
        freq = np.clip(label_freq, 1, None)
        effective_num = 1.0 - np.power(beta, freq)
        weights = (1.0 - beta) / effective_num
        weights = weights / (weights.mean() + 1e-12)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.unsqueeze(0).expand_as(logits)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="mean")


def build_loss(cfg: Config, label_freq: np.ndarray) -> nn.Module:
    if cfg.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if cfg.loss == "focal":
        return FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
    if cfg.loss == "cb_bce":
        return ClassBalancedBCE(label_freq=label_freq, beta=cfg.cb_beta)
    raise ValueError(f"Unknown loss: {cfg.loss}")


# =============================================================================
# Metrics
# =============================================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def micro_metrics_from_sets(y_true: List[List[str]], y_pred: List[List[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = set(t), set(p)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def decode_probs(probs: np.ndarray, label_list: List[str], threshold: Optional[float], cfg: Config) -> List[List[str]]:
    out = []
    for row in probs:
        pairs = sorted(zip(label_list, row.tolist()), key=lambda x: x[1], reverse=True)
        if threshold is None:
            chosen = [lab for lab, _ in pairs[:cfg.max_labels]]
        else:
            chosen = [lab for lab, p in pairs if p >= threshold]
            if len(chosen) < cfg.min_labels and pairs:
                chosen = [pairs[0][0]]
            if len(chosen) > cfg.max_labels:
                chosen = [lab for lab, _ in pairs[:cfg.max_labels]]
        out.append(dedupe_preserve_order(chosen))
    return out


def calibrate_threshold(probs: np.ndarray, gold: List[List[str]], label_list: List[str], cfg: Config) -> Tuple[float, Dict[str, Any]]:
    best_t, best_f1 = 0.5, -1.0
    records = []
    for t in parse_threshold_grid(cfg.threshold_grid):
        pred = decode_probs(probs, label_list, t, cfg)
        p, r, f1 = micro_metrics_from_sets(gold, pred)
        records.append({"threshold": t, "micro_precision": p, "micro_recall": r, "micro_f1": f1})
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, {"best_threshold": best_t, "best_micro_f1": best_f1, "grid": records}


def build_hier_tags(labels: List[str]) -> Set[str]:
    tags = set()
    for lab in labels:
        c = normalize_cpc_label(lab)
        if len(c) >= 1 and c[0].isalpha():
            tags.add(f"SECTION_{c[0]}")
        if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit():
            tags.add(f"CLASS_{c[:3]}")
        if len(c) >= 4:
            tags.add(f"SUBCLASS_{c[:4]}")
    return tags


def hierarchical_micro_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = build_hier_tags(t), build_hier_tags(p)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"hier_precision": float(prec), "hier_recall": float(rec), "hier_f1": float(f1)}


def evaluate_predictions(gold: List[List[str]], pred: List[List[str]], label_space: List[str]) -> Dict[str, Any]:
    mlb = MultiLabelBinarizer(classes=label_space)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        **hierarchical_micro_metrics(gold, pred),
        "avg_gold_labels": float(np.mean([len(x) for x in gold])) if gold else 0.0,
        "avg_pred_labels": float(np.mean([len(x) for x in pred])) if pred else 0.0,
        "empty_prediction_rate": float(np.mean([1.0 if not x else 0.0 for x in pred])) if pred else 0.0,
    }


def evaluate_rare_frequency_splits(gold: List[List[str]], pred: List[List[str]], label_list: List[str], train_freq: np.ndarray, cfg: Config) -> Dict[str, Any]:
    rare_idx = np.where(train_freq <= cfg.rare_threshold_freq)[0]
    freq_idx = np.where(train_freq > cfg.rare_threshold_freq)[0]

    mlb = MultiLabelBinarizer(classes=label_list)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(pred)

    def subset_macro(idx: np.ndarray) -> Tuple[float, int]:
        if len(idx) == 0:
            return 0.0, 0
        _, _, f1s, _ = precision_recall_fscore_support(y_true[:, idx], y_pred[:, idx], average=None, zero_division=0)
        return float(np.mean(f1s)), int(len(idx))

    rare_macro, n_rare = subset_macro(rare_idx)
    frequent_macro, n_frequent = subset_macro(freq_idx)
    return {
        "rare_threshold_freq": cfg.rare_threshold_freq,
        "macro_f1_rare": rare_macro,
        "macro_f1_frequent": frequent_macro,
        "n_rare_labels": n_rare,
        "n_frequent_labels": n_frequent,
    }


# =============================================================================
# Forward and training
# =============================================================================

@torch.no_grad()
def forward_probs(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_all = []
    for batch in tqdm(loader, desc="forward", leave=False):
        kwargs = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
        out = model(**kwargs)
        logits_all.append(out.logits.detach().cpu().numpy())
    return sigmoid_np(np.concatenate(logits_all, axis=0))


def save_checkpoint(model, tokenizer, out_dir: Path, name: str) -> Path:
    ckpt_dir = out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    return ckpt_dir


def train(cfg: Config) -> None:
    out_dir = Path(cfg.output_dir)
    label_list = load_label_list(cfg.label_mapping)
    label_set = set(label_list)

    train_texts, train_labels, _, _ = load_split(cfg.train_path, cfg, label_set, cfg.max_train_examples)
    dev_texts, dev_labels, _, _ = load_split(cfg.dev_path, cfg, label_set, cfg.max_dev_examples)
    test_texts, test_labels, test_titles, _ = load_split(cfg.test_path, cfg, label_set, cfg.max_test_examples)

    train_freq = compute_label_frequencies(train_labels, label_list)

    if cfg.eval_label_space_from_mapping_only:
        eval_label_space = list(label_list)
        unseen_test = sorted({lab for labs in test_labels for lab in labs if lab not in label_set})
    else:
        eval_label_space = sorted(set(label_list) | {lab for labs in test_labels for lab in labs})
        unseen_test = []

    save_json(out_dir / "data_summary.json", {
        "n_train": len(train_texts),
        "n_dev": len(dev_texts),
        "n_test": len(test_texts),
        "n_labels": len(label_list),
        "avg_train_labels": float(np.mean([len(x) for x in train_labels])),
        "avg_dev_labels": float(np.mean([len(x) for x in dev_labels])),
        "avg_test_labels": float(np.mean([len(x) for x in test_labels])),
        "rare_threshold_freq": cfg.rare_threshold_freq,
        "n_rare_labels_train": int(np.sum(train_freq <= cfg.rare_threshold_freq)),
        "n_frequent_labels_train": int(np.sum(train_freq > cfg.rare_threshold_freq)),
        "unseen_test_labels_ignored": unseen_test,
    })

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=False, local_files_only=cfg.offline)
    train_ds = PatentDataset(train_texts, train_labels, label_list, tokenizer, cfg.max_length)
    dev_ds = PatentDataset(dev_texts, dev_labels, label_list, tokenizer, cfg.max_length)
    test_ds = PatentDataset(test_texts, test_labels, label_list, tokenizer, cfg.max_length)

    train_loader = make_train_loader(train_ds, train_labels, label_list, train_freq, cfg)
    dev_loader = make_eval_loader(dev_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        num_labels=len(label_list),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        local_files_only=cfg.offline,
    )
    model.config.id2label = {i: lab for i, lab in enumerate(label_list)}
    model.config.label2id = {lab: i for i, lab in enumerate(label_list)}
    model.to(device)

    loss_fn = build_loss(cfg, train_freq).to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=cfg.learning_rate)

    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = update_steps_per_epoch * cfg.max_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    tracker = None
    emissions_kg = None
    if cfg.use_codecarbon:
        if EmissionsTracker is None:
            print("[warn] codecarbon unavailable; emissions tracking disabled.")
        else:
            tracker = EmissionsTracker(
                project_name=cfg.codecarbon_project_name,
                output_dir=str(out_dir),
                measure_power_secs=cfg.codecarbon_measure_power_secs,
                save_to_file=True,
                log_level="error",
            )

    best_dev_f1 = -1.0
    best_ckpt: Optional[Path] = None
    patience_counter = 0
    global_step = 0
    training_log = []
    start_time = time.time()

    if tracker:
        tracker.start()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False), start=1):
            kwargs = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(**kwargs)
            loss = loss_fn(out.logits, labels) / cfg.grad_accum
            loss.backward()
            epoch_loss += float(loss.item()) * cfg.grad_accum

            if step % cfg.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        dev_probs = forward_probs(model, dev_loader, device)
        dev_threshold, threshold_info = calibrate_threshold(dev_probs, dev_labels, label_list, cfg)
        dev_pred = decode_probs(dev_probs, label_list, dev_threshold, cfg)
        _, _, dev_micro_f1 = micro_metrics_from_sets(dev_labels, dev_pred)
        avg_loss = epoch_loss / max(len(train_loader), 1)

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_loss,
            "dev_micro_f1": dev_micro_f1,
            "dev_threshold": dev_threshold,
            "threshold_info": threshold_info,
        }
        training_log.append(row)
        save_json(out_dir / "training_log.json", training_log)

        print(f"epoch={epoch} loss={avg_loss:.4f} dev_micro_f1={dev_micro_f1:.4f} threshold={dev_threshold:.2f}")

        if cfg.save_all_epoch_checkpoints:
            save_checkpoint(model, tokenizer, out_dir, f"checkpoint-{global_step}")

        if dev_micro_f1 > best_dev_f1:
            best_dev_f1 = dev_micro_f1
            patience_counter = 0
            best_ckpt = save_checkpoint(model, tokenizer, out_dir, "best_checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("early stopping triggered")
                break

    if tracker:
        emissions_kg = tracker.stop()

    total_time_sec = time.time() - start_time
    if best_ckpt is None:
        best_ckpt = save_checkpoint(model, tokenizer, out_dir, "best_checkpoint")

    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_ckpt), local_files_only=True).to(device)
    best_model.eval()

    dev_probs = forward_probs(best_model, dev_loader, device)
    final_threshold, threshold_info = calibrate_threshold(dev_probs, dev_labels, label_list, cfg)
    test_probs = forward_probs(best_model, test_loader, device)
    test_pred = decode_probs(test_probs, label_list, final_threshold, cfg)

    metrics = evaluate_predictions(test_labels, test_pred, eval_label_space)
    rare_metrics = evaluate_rare_frequency_splits(test_labels, test_pred, label_list, train_freq, cfg)
    metrics.update(rare_metrics)

    final_summary = {
        "model_name": cfg.model_name,
        "model_path": cfg.model_path,
        "best_checkpoint": str(best_ckpt),
        "best_dev_micro_f1": best_dev_f1,
        "final_dev_threshold": final_threshold,
        "threshold_info": threshold_info,
        "total_train_time_sec": total_time_sec,
        "total_emissions_kg": emissions_kg,
        "test_metrics": metrics,
    }
    save_json(out_dir / "test_results.json", final_summary)

    with (out_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, (title, text, gold, pred) in enumerate(zip(test_titles, test_texts, test_labels, test_pred)):
            f.write(json.dumps({"idx": i, "title": title, "text": text, "gold": gold, "pred": pred}, ensure_ascii=False) + "\n")

    print("\n=== Final test metrics ===")
    print(json.dumps(metrics, indent=2))

    del model, best_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    configure_environment(cfg)
    set_all_seeds(cfg.seed)

    out_dir = Path(cfg.output_dir)
    prepare_output_dir(out_dir, cfg.overwrite)
    save_json(out_dir / "run_environment.json", environment_report(cfg))

    train(cfg)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
