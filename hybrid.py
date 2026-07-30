#!/usr/bin/env python3
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
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}")
_JSON_LIST_RE = re.compile(r"\[[\s\S]*?\]")
_CPC_RE = re.compile(r"\b[A-HY]\d{2}[A-Z]\b")

SYSTEM_PROMPT = (
    "You are an expert patent examiner specialized in Cooperative Patent Classification "
    "(CPC) subclass assignment. You are correcting the output of an encoder classifier. "
    "Use the patent text, encoder predictions, encoder candidate probabilities, and CPC "
    "definitions. Return ONLY valid JSON with exactly one key: \"labels\". The value must "
    "be a list of CPC subclass codes. Do not explain. Prefer precision over recall. Keep "
    "encoder labels unless clearly unsupported. Add a new label only when strongly supported "
    "by the patent text and CPC definition."
)

USER_PROMPT_TEMPLATE = """PATENT TEXT:
----------------------------------------
Title: {title}
Abstract: {abstract}
----------------------------------------

ENCODER CURRENT PREDICTION:
{encoder_predictions_block}

ENCODER TOP CANDIDATES:
{encoder_candidates_block}

CPC DEFINITIONS:
{cpc_definitions_block}

ALLOWED OUTPUT LABELS:
{allowed_labels_block}

TASK:
Correct the encoder prediction.
- Output the final CPC subclass labels for this patent.
- Use only labels from ALLOWED OUTPUT LABELS.
- Keep encoder labels unless clearly wrong.
- Add labels only if strongly supported by the patent text and CPC definition.
- Return 1 to {max_labels} CPC subclass codes.
- Return JSON only.

OUTPUT FORMAT:
{{"labels": ["G06F", "H04L"]}}
"""


@dataclass
class Config:
    train_path: str
    dev_path: str
    test_path: str
    label_mapping: str
    encoder_checkpoint: str
    encoder_tokenizer: str
    llm_base: str
    llm_lora: str
    cpc_definitions: str
    out_dir: str
    input_format: str = "auto"
    title_column: str = "title"
    abstract_column: str = "abstract"
    labels_column: str = "labels"
    seed: int = 42
    encoder_max_length: int = 256
    encoder_batch_size: int = 16
    llm_batch_size: int = 4
    llm_context_length: int = 2048
    llm_max_new_tokens: int = 64
    min_labels: int = 1
    max_labels: int = 7
    threshold_grid: List[float] = field(default_factory=lambda: [round(float(x), 4) for x in np.linspace(0.05, 0.95, 37)])
    routing_fractions: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.20])
    uncertainty_methods: List[str] = field(default_factory=lambda: ["maxprob", "margin", "entropy"])
    selection_metric: str = "macro_f1"
    encoder_topk_candidates: int = 25
    max_candidates_in_prompt: int = 25
    strict_json: bool = True
    save_prompts: bool = True
    save_predictions: bool = True
    overwrite: bool = False
    offline: bool = False


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-selected encoder-LLM hybrid routing for CPC classification.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--dev-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--label-mapping", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--encoder-tokenizer", required=True)
    parser.add_argument("--llm-base", required=True)
    parser.add_argument("--llm-lora", required=True)
    parser.add_argument("--cpc-definitions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-format", default="auto", choices=["auto", "csv", "tsv", "json", "jsonl"])
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--abstract-column", default="abstract")
    parser.add_argument("--labels-column", default="labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-context-length", type=int, default=2048)
    parser.add_argument("--llm-max-new-tokens", type=int, default=64)
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--max-labels", type=int, default=7)
    parser.add_argument("--routing-fractions", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--uncertainty-methods", default="maxprob,margin,entropy")
    parser.add_argument("--selection-metric", default="macro_f1", choices=["micro_f1", "macro_f1", "hier_f1", "routed_micro_delta", "routed_macro_delta", "routed_hier_delta"])
    parser.add_argument("--encoder-topk-candidates", type=int, default=25)
    parser.add_argument("--max-candidates-in-prompt", type=int, default=25)
    parser.add_argument("--allow-regex-fallback", action="store_true")
    parser.add_argument("--no-save-prompts", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        label_mapping=args.label_mapping,
        encoder_checkpoint=args.encoder_checkpoint,
        encoder_tokenizer=args.encoder_tokenizer,
        llm_base=args.llm_base,
        llm_lora=args.llm_lora,
        cpc_definitions=args.cpc_definitions,
        out_dir=args.out_dir,
        input_format=args.input_format,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        labels_column=args.labels_column,
        seed=args.seed,
        encoder_max_length=args.encoder_max_length,
        encoder_batch_size=args.encoder_batch_size,
        llm_batch_size=args.llm_batch_size,
        llm_context_length=args.llm_context_length,
        llm_max_new_tokens=args.llm_max_new_tokens,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        routing_fractions=parse_float_list(args.routing_fractions),
        uncertainty_methods=parse_str_list(args.uncertainty_methods),
        selection_metric=args.selection_metric,
        encoder_topk_candidates=args.encoder_topk_candidates,
        max_candidates_in_prompt=args.max_candidates_in_prompt,
        strict_json=not args.allow_regex_fallback,
        save_prompts=not args.no_save_prompts,
        save_predictions=not args.no_save_predictions,
        overwrite=args.overwrite,
        offline=args.offline,
    )


def configure_environment(cfg: Config) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_cpc(value: Any) -> str:
    code = str(value or "").strip().upper().split("/")[0]
    code = re.sub(r"[^A-Z0-9]", "", code)
    if len(code) >= 4 and code[0].isalpha() and code[1:3].isdigit() and code[3].isalpha():
        return code[:4]
    return ""


def dedupe(labels: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in labels:
        code = normalize_cpc(value)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def parse_labels(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return dedupe(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, dict):
                    parsed = parsed.get("labels")
                if isinstance(parsed, (list, tuple, set)):
                    return dedupe(parsed)
            except Exception:
                pass
        return dedupe(re.split(r"[,;|]", value))
    return dedupe([raw])


def infer_format(path: str, requested: str) -> Tuple[str, Optional[str]]:
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
    raise ValueError(f"Cannot infer input format from {path}")


def load_split(path: str, cfg: Config) -> Dict[str, Any]:
    dataset_type, delimiter = infer_format(path, cfg.input_format)
    kwargs: Dict[str, Any] = {"data_files": {"data": path}}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    dataset = load_dataset(dataset_type, **kwargs)["data"]
    required = [cfg.title_column, cfg.abstract_column, cfg.labels_column]
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    titles: List[str] = []
    abstracts: List[str] = []
    texts: List[str] = []
    labels: List[List[str]] = []
    for row in dataset:
        title = str(row.get(cfg.title_column) or "").strip()
        abstract = str(row.get(cfg.abstract_column) or "").strip()
        text = f"{title}. {abstract}".strip(". ")
        row_labels = parse_labels(row.get(cfg.labels_column))
        if text and row_labels:
            titles.append(title)
            abstracts.append(abstract)
            texts.append(text)
            labels.append(row_labels)
    return {"titles": titles, "abstracts": abstracts, "texts": texts, "gold": labels}


def load_label_mapping(path: str) -> List[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        values = obj
    elif isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        values = obj["labels"]
    elif isinstance(obj, dict) and isinstance(obj.get("label2id"), dict):
        values = [key for key, value in sorted(obj["label2id"].items(), key=lambda item: int(item[1]))]
    elif isinstance(obj, dict) and isinstance(obj.get("id2label"), dict):
        values = [obj["id2label"][str(i)] for i in sorted(map(int, obj["id2label"]))]
    else:
        raise ValueError("Unsupported label mapping format")
    labels = dedupe(values)
    if not labels:
        raise ValueError("No labels found in label mapping")
    return labels


def load_cpc_definitions(path: str, valid_labels: Set[str]) -> Dict[str, str]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    out: Dict[str, str] = {}

    def add(code: Any, definition: Any) -> None:
        normalized = normalize_cpc(code)
        text = str(definition or "").strip()
        if normalized in valid_labels and text:
            out[normalized] = text

    if file_path.suffix.lower() == ".json":
        obj = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    definition = value.get("definition") or value.get("description") or value.get("text") or value.get("title")
                else:
                    definition = value
                add(key, definition)
        elif isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    code = row.get("code") or row.get("label") or row.get("cpc") or row.get("subclass")
                    definition = row.get("definition") or row.get("description") or row.get("text") or row.get("title")
                    add(code, definition)
    elif file_path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                lowered = {str(key).lower(): value for key, value in row.items()}
                code = lowered.get("code") or lowered.get("label") or lowered.get("cpc") or lowered.get("subclass")
                definition = lowered.get("definition") or lowered.get("description") or lowered.get("text") or lowered.get("title")
                add(code, definition)
    else:
        raise ValueError("Unsupported CPC definitions format")
    return out


def environment_report(cfg: Config) -> Dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": asdict(cfg),
    }


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def make_dataloader(texts: List[str], tokenizer, batch_size: int, max_length: int) -> DataLoader:
    dataset = TextDataset(texts)

    def collate(batch: List[str]):
        return tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def load_encoder(checkpoint: str, tokenizer_path: str, labels: List[str]):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir() or not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid encoder checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.config.id2label = {i: label for i, label in enumerate(labels)}
    model.config.label2id = {label: i for i, label in enumerate(labels)}
    out_features = getattr(getattr(model, "classifier", None), "out_features", None)
    if out_features is not None and out_features != len(labels):
        raise ValueError(f"Encoder output size {out_features} does not match label mapping size {len(labels)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def encoder_probabilities(model, tokenizer, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    loader = make_dataloader(texts, tokenizer, batch_size, max_length)
    device = next(model.parameters()).device
    logits: List[np.ndarray] = []
    for batch in tqdm(loader, desc="Encoder inference"):
        batch = {key: value.to(device) for key, value in batch.items()}
        logits.append(model(**batch).logits.detach().cpu().numpy())
    values = np.concatenate(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


def decode_probabilities(probs: np.ndarray, labels: List[str], threshold: float, min_labels: int, max_labels: int) -> List[List[str]]:
    output: List[List[str]] = []
    for row in probs:
        ranked = sorted(zip(labels, row.tolist()), key=lambda item: item[1], reverse=True)
        selected = [label for label, score in ranked if score >= threshold]
        if len(selected) < min_labels:
            selected = [ranked[0][0]]
        if len(selected) > max_labels:
            selected = [label for label, _ in ranked[:max_labels]]
        output.append(selected)
    return output


def topk_candidates(probs: np.ndarray, labels: List[str], k: int) -> List[List[Tuple[str, float]]]:
    output: List[List[Tuple[str, float]]] = []
    for row in probs:
        ranked = sorted(zip(labels, row.tolist()), key=lambda item: item[1], reverse=True)[:k]
        output.append([(label, float(score)) for label, score in ranked])
    return output


def micro_f1_sets(gold: List[List[str]], pred: List[List[str]]) -> float:
    tp = fp = fn = 0
    for gold_row, pred_row in zip(gold, pred):
        gold_set = set(gold_row)
        pred_set = set(pred_row)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def calibrate_threshold(dev_probs: np.ndarray, dev_gold: List[List[str]], labels: List[str], cfg: Config) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in cfg.threshold_grid:
        pred = decode_probabilities(dev_probs, labels, threshold, cfg.min_labels, cfg.max_labels)
        score = micro_f1_sets(dev_gold, pred)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return float(best_threshold)


def build_hierarchy(labels: List[str]) -> Set[str]:
    output: Set[str] = set()
    for label in labels:
        code = normalize_cpc(label)
        if code:
            output.add(f"SECTION_{code[0]}")
            output.add(f"CLASS_{code[:3]}")
            output.add(f"SUBCLASS_{code}")
    return output


def hierarchical_metrics(gold: List[List[str]], pred: List[List[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for gold_row, pred_row in zip(gold, pred):
        gold_set = build_hierarchy(gold_row)
        pred_set = build_hierarchy(pred_row)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(gold: List[List[str]], pred: List[List[str]], label_space: List[str]) -> Dict[str, Any]:
    mlb = MultiLabelBinarizer(classes=label_space)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(pred)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    h_precision, h_recall, h_f1 = hierarchical_metrics(gold, pred)
    return {
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "hierarchical_precision": float(h_precision),
        "hierarchical_recall": float(h_recall),
        "hierarchical_f1": float(h_f1),
        "avg_gold_labels": float(np.mean([len(row) for row in gold])),
        "avg_pred_labels": float(np.mean([len(row) for row in pred])),
    }


def training_frequency_groups(train_gold: List[List[str]], label_space: List[str]) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    frequencies = Counter(label for row in train_gold for label in row)
    ordered = sorted(label_space, key=lambda label: frequencies.get(label, 0))
    count = max(1, int(math.ceil(0.20 * len(ordered))))
    rare = set(ordered[:count])
    frequent = set(ordered[-count:])
    return rare, frequent, {label: int(frequencies.get(label, 0)) for label in label_space}


def evaluate_label_subset(gold: List[List[str]], pred: List[List[str]], labels: Set[str]) -> Dict[str, float]:
    selected = sorted(labels)
    if not selected:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "n_labels": 0}
    filtered_gold = [[label for label in row if label in labels] for row in gold]
    filtered_pred = [[label for label in row if label in labels] for row in pred]
    mlb = MultiLabelBinarizer(classes=selected)
    y_true = mlb.fit_transform(filtered_gold)
    y_pred = mlb.transform(filtered_pred)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"micro_f1": float(micro[2]), "macro_f1": float(macro[2]), "n_labels": len(selected)}


def compute_uncertainty(probs: np.ndarray, method: str) -> np.ndarray:
    if method == "maxprob":
        return 1.0 - probs.max(axis=1)
    if method == "margin":
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        return 1.0 - (np.max(top2, axis=1) - np.min(top2, axis=1))
    if method == "entropy":
        clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0 - 1e-12)
        return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)).sum(axis=1)
    raise ValueError(f"Unknown uncertainty method: {method}")


def load_llm(base_path: str, lora_path: str):
    if PeftModel is None:
        raise ImportError("peft is required")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto", torch_dtype=dtype).eval()
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False).eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def truncate_text(value: Any, length: int) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    return text if len(text) <= length else text[: length - 3] + "..."


def format_encoder_prediction(labels: List[str], scores: Dict[str, float]) -> str:
    return "\n".join(f"- {label} | encoder_prob={scores.get(label, 0.0):.4f}" for label in labels) or "- None"


def format_encoder_candidates(candidates: List[Tuple[str, float]], definitions: Dict[str, str], max_items: int) -> str:
    lines = []
    for label, score in candidates[:max_items]:
        definition = truncate_text(definitions.get(label, ""), 180)
        lines.append(f"- {label} | encoder_prob={score:.4f} | definition={definition}")
    return "\n".join(lines) or "- None"


def format_definitions(candidates: List[Tuple[str, float]], definitions: Dict[str, str], max_items: int) -> str:
    lines = []
    for label, _ in candidates[:max_items]:
        lines.append(f"- {label} --- {truncate_text(definitions.get(label, ''), 220)}")
    return "\n".join(lines) or "- None"


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    for extra in ({"enable_thinking": False}, {"chat_template_kwargs": {"enable_thinking": False}}, {}):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        except TypeError:
            pass
    return f"[SYSTEM]\n{messages[0]['content']}\n\n[USER]\n{messages[1]['content']}\n\n[ASSISTANT]\n"


def build_prompt(tokenizer, cfg: Config, title: str, abstract: str, encoder_pred: List[str], candidates: List[Tuple[str, float]], definitions: Dict[str, str]) -> str:
    scores = {label: score for label, score in candidates}
    allowed = dedupe(encoder_pred + [label for label, _ in candidates[: cfg.encoder_topk_candidates]])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=truncate_text(title, 260),
        abstract=truncate_text(abstract, 1800),
        encoder_predictions_block=format_encoder_prediction(encoder_pred, scores),
        encoder_candidates_block=format_encoder_candidates(candidates, definitions, cfg.max_candidates_in_prompt),
        cpc_definitions_block=format_definitions(candidates, definitions, cfg.max_candidates_in_prompt),
        allowed_labels_block=", ".join(allowed),
        max_labels=cfg.max_labels,
    )
    return apply_chat_template(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])


def filter_labels(values: Iterable[Any], allowed: Set[str], max_labels: int) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        code = normalize_cpc(value)
        if code and code in allowed and code not in seen:
            seen.add(code)
            output.append(code)
        if len(output) >= max_labels:
            break
    return output


def parse_llm_output(text: str, allowed: Set[str], cfg: Config) -> Tuple[List[str], str]:
    raw = (text or "").strip()
    if not raw:
        return [], "empty"

    def extract(obj: Any) -> List[str]:
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return filter_labels(obj["labels"], allowed, cfg.max_labels)
        if isinstance(obj, list):
            return filter_labels(obj, allowed, cfg.max_labels)
        return []

    try:
        labels = extract(json.loads(raw))
        if labels:
            return labels, "strict_json"
    except Exception:
        pass

    match = _JSON_OBJ_RE.search(raw)
    if match:
        try:
            labels = extract(json.loads(match.group(0)))
            if labels:
                return labels, "embedded_json"
        except Exception:
            pass

    match = _JSON_LIST_RE.search(raw)
    if match:
        try:
            labels = extract(json.loads(match.group(0)))
            if labels:
                return labels, "embedded_json_list"
        except Exception:
            pass

    if not cfg.strict_json:
        labels = filter_labels(_CPC_RE.findall(raw.upper()), allowed, cfg.max_labels)
        if labels:
            return labels, "regex"

    return [], "parse_failed"


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], cfg: Config) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.llm_context_length)
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.llm_max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = inputs["input_ids"].shape[1]
    return [tokenizer.decode(outputs[i, prompt_width:], skip_special_tokens=True).strip() for i in range(len(prompts))]


def run_llm_on_indices(
    model,
    tokenizer,
    cfg: Config,
    titles: List[str],
    abstracts: List[str],
    encoder_preds: List[List[str]],
    encoder_candidates: List[List[Tuple[str, float]]],
    definitions: Dict[str, str],
    indices: List[int],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for start in tqdm(range(0, len(indices), cfg.llm_batch_size), desc="LLM correction"):
        batch_indices = indices[start : start + cfg.llm_batch_size]
        prompts = [
            build_prompt(
                tokenizer,
                cfg,
                titles[index],
                abstracts[index],
                encoder_preds[index],
                encoder_candidates[index],
                definitions,
            )
            for index in batch_indices
        ]
        outputs = generate_batch(model, tokenizer, prompts, cfg)
        for index, prompt, raw in zip(batch_indices, prompts, outputs):
            allowed = set(dedupe(encoder_preds[index] + [label for label, _ in encoder_candidates[index][: cfg.encoder_topk_candidates]]))
            labels, parse_mode = parse_llm_output(raw, allowed, cfg)
            fallback = False
            if not labels:
                labels = list(encoder_preds[index])
                fallback = True
            results[index] = {
                "pred": labels[: cfg.max_labels],
                "raw_generation": raw,
                "parse_mode": parse_mode,
                "fallback_to_encoder": fallback,
                "prompt": prompt if cfg.save_prompts else None,
            }
    return results


def evaluate_routed_subset(gold: List[List[str]], encoder_pred: List[List[str]], hybrid_pred: List[List[str]], indices: List[int], label_space: List[str]) -> Dict[str, Any]:
    routed_gold = [gold[i] for i in indices]
    routed_encoder = [encoder_pred[i] for i in indices]
    routed_hybrid = [hybrid_pred[i] for i in indices]
    encoder_metrics = evaluate(routed_gold, routed_encoder, label_space)
    hybrid_metrics = evaluate(routed_gold, routed_hybrid, label_space)
    return {
        "n_routed": len(indices),
        "encoder": encoder_metrics,
        "hybrid": hybrid_metrics,
        "delta": {
            "micro_f1": hybrid_metrics["micro_f1"] - encoder_metrics["micro_f1"],
            "macro_f1": hybrid_metrics["macro_f1"] - encoder_metrics["macro_f1"],
            "hier_f1": hybrid_metrics["hierarchical_f1"] - encoder_metrics["hierarchical_f1"],
        },
    }


def evaluate_configuration(
    split_name: str,
    cfg: Config,
    probs: np.ndarray,
    gold: List[List[str]],
    titles: List[str],
    abstracts: List[str],
    encoder_pred: List[List[str]],
    encoder_candidates: List[List[Tuple[str, float]]],
    definitions: Dict[str, str],
    label_space: List[str],
    rare_labels: Set[str],
    frequent_labels: Set[str],
    llm_model,
    llm_tokenizer,
    uncertainty_method: str,
    routing_fraction: float,
    out_dir: Path,
) -> Dict[str, Any]:
    scores = compute_uncertainty(probs, uncertainty_method)
    n_route = max(1, int(round(routing_fraction * len(gold))))
    route_indices = np.argsort(-scores)[:n_route].tolist()
    route_set = set(route_indices)
    llm_outputs = run_llm_on_indices(
        llm_model,
        llm_tokenizer,
        cfg,
        titles,
        abstracts,
        encoder_pred,
        encoder_candidates,
        definitions,
        route_indices,
    )
    hybrid_pred = [
        llm_outputs[i]["pred"] if i in route_set else encoder_pred[i]
        for i in range(len(gold))
    ]
    encoder_metrics = evaluate(gold, encoder_pred, label_space)
    hybrid_metrics = evaluate(gold, hybrid_pred, label_space)
    routed_metrics = evaluate_routed_subset(gold, encoder_pred, hybrid_pred, route_indices, label_space)
    result = {
        "split": split_name,
        "uncertainty_method": uncertainty_method,
        "routing_fraction": routing_fraction,
        "num_routed": n_route,
        "encoder_metrics": encoder_metrics,
        "hybrid_metrics": hybrid_metrics,
        "full_dataset_delta": {
            "micro_f1": hybrid_metrics["micro_f1"] - encoder_metrics["micro_f1"],
            "macro_f1": hybrid_metrics["macro_f1"] - encoder_metrics["macro_f1"],
            "hier_f1": hybrid_metrics["hierarchical_f1"] - encoder_metrics["hierarchical_f1"],
        },
        "routed_subset": routed_metrics,
        "rare_labels": {
            "encoder": evaluate_label_subset(gold, encoder_pred, rare_labels),
            "hybrid": evaluate_label_subset(gold, hybrid_pred, rare_labels),
        },
        "frequent_labels": {
            "encoder": evaluate_label_subset(gold, encoder_pred, frequent_labels),
            "hybrid": evaluate_label_subset(gold, hybrid_pred, frequent_labels),
        },
        "parse_diagnostics": {
            "fallback_count": sum(int(record["fallback_to_encoder"]) for record in llm_outputs.values()),
            "fallback_rate": float(np.mean([record["fallback_to_encoder"] for record in llm_outputs.values()])),
            "parse_modes": dict(Counter(record["parse_mode"] for record in llm_outputs.values())),
        },
    }
    tag = f"{split_name}_{uncertainty_method}_{int(round(100 * routing_fraction)):02d}pct"
    save_json(out_dir / f"metrics_{tag}.json", result)
    if cfg.save_predictions:
        with (out_dir / f"predictions_{tag}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(len(gold)):
                record = {
                    "idx": index,
                    "title": titles[index],
                    "abstract": abstracts[index],
                    "gold": gold[index],
                    "encoder_pred": encoder_pred[index],
                    "encoder_candidates": encoder_candidates[index],
                    "hybrid_pred": hybrid_pred[index],
                    "routed": index in route_set,
                    "uncertainty_score": float(scores[index]),
                }
                if index in route_set:
                    record["llm"] = llm_outputs[index]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def selection_score(result: Dict[str, Any], metric: str) -> Tuple[float, float]:
    hybrid = result["hybrid_metrics"]
    routed_delta = result["routed_subset"]["delta"]
    if metric == "micro_f1":
        return hybrid["micro_f1"], hybrid["macro_f1"]
    if metric == "macro_f1":
        return hybrid["macro_f1"], hybrid["micro_f1"]
    if metric == "hier_f1":
        return hybrid["hierarchical_f1"], hybrid["macro_f1"]
    if metric == "routed_micro_delta":
        return routed_delta["micro_f1"], hybrid["micro_f1"]
    if metric == "routed_macro_delta":
        return routed_delta["macro_f1"], hybrid["macro_f1"]
    if metric == "routed_hier_delta":
        return routed_delta["hier_f1"], hybrid["hierarchical_f1"]
    raise ValueError(metric)


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    configure_environment(cfg)
    set_seeds(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out_dir = Path(cfg.out_dir)
    prepare_output_dir(out_dir, cfg.overwrite)
    save_json(out_dir / "run_environment.json", environment_report(cfg))

    label_space = load_label_mapping(cfg.label_mapping)
    label_set = set(label_space)
    definitions = load_cpc_definitions(cfg.cpc_definitions, label_set)

    train = load_split(cfg.train_path, cfg)
    dev = load_split(cfg.dev_path, cfg)
    test = load_split(cfg.test_path, cfg)

    rare_labels, frequent_labels, train_frequencies = training_frequency_groups(train["gold"], label_space)
    save_json(
        out_dir / "data_summary.json",
        {
            "n_train": len(train["gold"]),
            "n_dev": len(dev["gold"]),
            "n_test": len(test["gold"]),
            "n_labels": len(label_space),
            "n_rare_labels": len(rare_labels),
            "n_frequent_labels": len(frequent_labels),
            "training_frequencies": train_frequencies,
        },
    )

    encoder_model, encoder_tokenizer = load_encoder(cfg.encoder_checkpoint, cfg.encoder_tokenizer, label_space)
    dev_probs = encoder_probabilities(encoder_model, encoder_tokenizer, dev["texts"], cfg.encoder_batch_size, cfg.encoder_max_length)
    test_probs = encoder_probabilities(encoder_model, encoder_tokenizer, test["texts"], cfg.encoder_batch_size, cfg.encoder_max_length)

    threshold = calibrate_threshold(dev_probs, dev["gold"], label_space, cfg)
    dev_encoder_pred = decode_probabilities(dev_probs, label_space, threshold, cfg.min_labels, cfg.max_labels)
    test_encoder_pred = decode_probabilities(test_probs, label_space, threshold, cfg.min_labels, cfg.max_labels)
    dev_candidates = topk_candidates(dev_probs, label_space, cfg.encoder_topk_candidates)
    test_candidates = topk_candidates(test_probs, label_space, cfg.encoder_topk_candidates)

    save_json(
        out_dir / "encoder_baseline.json",
        {
            "threshold": threshold,
            "dev_metrics": evaluate(dev["gold"], dev_encoder_pred, label_space),
            "test_metrics": evaluate(test["gold"], test_encoder_pred, label_space),
        },
    )

    llm_model, llm_tokenizer = load_llm(cfg.llm_base, cfg.llm_lora)

    dev_results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score = (-math.inf, -math.inf)

    for method in cfg.uncertainty_methods:
        for fraction in cfg.routing_fractions:
            result = evaluate_configuration(
                "dev",
                cfg,
                dev_probs,
                dev["gold"],
                dev["titles"],
                dev["abstracts"],
                dev_encoder_pred,
                dev_candidates,
                definitions,
                label_space,
                rare_labels,
                frequent_labels,
                llm_model,
                llm_tokenizer,
                method,
                fraction,
                out_dir,
            )
            dev_results.append(result)
            score = selection_score(result, cfg.selection_metric)
            if score > best_score:
                best_score = score
                best_result = result

    if best_result is None:
        raise RuntimeError("No hybrid configuration evaluated")

    selected = {
        "uncertainty_method": best_result["uncertainty_method"],
        "routing_fraction": best_result["routing_fraction"],
        "selection_metric": cfg.selection_metric,
        "selection_score": best_score,
    }
    save_json(out_dir / "dev_selection.json", {"selected": selected, "results": dev_results})

    test_result = evaluate_configuration(
        "test",
        cfg,
        test_probs,
        test["gold"],
        test["titles"],
        test["abstracts"],
        test_encoder_pred,
        test_candidates,
        definitions,
        label_space,
        rare_labels,
        frequent_labels,
        llm_model,
        llm_tokenizer,
        selected["uncertainty_method"],
        selected["routing_fraction"],
        out_dir,
    )

    save_json(
        out_dir / "final_result.json",
        {
            "encoder_threshold": threshold,
            "selected_on_dev": selected,
            "test_result": test_result,
        },
    )

    del llm_model, llm_tokenizer, encoder_model, encoder_tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
