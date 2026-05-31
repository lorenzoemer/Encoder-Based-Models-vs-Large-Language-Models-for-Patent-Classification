
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
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


# =============================================================================
# Constants
# =============================================================================

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}")
_JSON_LIST_RE = re.compile(r"\[[\s\S]*?\]")
_CPC_RE = re.compile(r"\b[A-HY]\d{2}[A-Z]\b")

SYSTEM_PROMPT_AUDIT = (
    "You are an expert patent examiner specialized in Cooperative Patent Classification "
    "(CPC) subclass assignment. You are correcting the output of an encoder classifier. "
    "Use the patent text, encoder predictions, encoder candidate probabilities, retrieval "
    "scores, and CPC definitions. Return ONLY valid JSON with exactly one key: \"labels\". "
    "The value must be a list of CPC subclass codes. Do not explain. Prefer precision over "
    "recall. Keep encoder labels unless clearly unsupported. Add a new label only when "
    "strongly supported by the patent text and CPC definition."
)

USER_PROMPT_AUDIT_TEMPLATE = """PATENT TEXT:
----------------------------------------
Title: {title}
Abstract: {abstract}
----------------------------------------

ENCODER CURRENT PREDICTION:
{encoder_predictions_block}

ENCODER TOP CANDIDATES:
{encoder_candidates_block}

RAG CPC CANDIDATES:
{rag_candidates_block}

ALLOWED OUTPUT LABELS:
{allowed_labels_block}

TASK:
Correct the encoder prediction.
- Output the final CPC subclass labels for this patent.
- Use only labels from ALLOWED OUTPUT LABELS.
- Keep encoder labels unless clearly wrong.
- Add labels only if strongly supported by the patent text and the CPC definition.
- Return 1 to {max_labels} CPC subclass codes.
- Return JSON only.

OUTPUT FORMAT:
{{"labels": ["G06F", "H04L"]}}
"""


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    dev_path: str
    test_path: str
    label_mapping: str
    encoder_run_dir: str
    encoder_tokenizer_path: str
    llm_base_path: str
    out_dir: str

    llm_lora_path: Optional[str] = None
    cpc_definitions_path: Optional[str] = None
    rag_model_path: Optional[str] = None
    no_rag: bool = False
    offline: bool = False
    overwrite: bool = False

    title_column: str = "title"
    abstract_column: str = "abstract"
    labels_column: str = "labels"
    input_format: str = "auto"  # auto, csv, tsv, json, jsonl

    seed: int = 1
    max_length: int = 256
    enc_batch_size: int = 16
    llm_batch_size: int = 4
    llm_max_new_tokens: int = 128
    llm_context_length: int = 2048

    min_labels: int = 1
    max_labels: int = 7
    threshold_grid: List[float] = field(default_factory=lambda: [round(float(x), 4) for x in np.linspace(0.05, 0.95, 37)])

    routing_fractions: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.15, 0.20])
    uncertainty_methods: List[str] = field(default_factory=lambda: ["maxprob", "margin", "entropy"])
    merge_rules: List[str] = field(default_factory=lambda: ["audit_final"])
    selection_metric: str = "macro_f1"  # micro_f1, macro_f1, hier_f1, routed_hier_delta, routed_macro_delta

    encoder_topk_candidates: int = 30
    rag_topk_candidates: int = 20
    max_allowed_candidates_for_llm: int = 45
    max_encoder_topk_in_prompt: int = 12
    max_rag_topk_in_prompt: int = 12

    add_min_encoder_prob_grid: List[float] = field(default_factory=lambda: [0.0])
    add_min_rag_score_grid: List[float] = field(default_factory=lambda: [0.0])

    allow_llm_only_from_allowed_candidates: bool = True
    strict_no_regex_fallback: bool = True
    save_prompts: bool = True
    save_predictions: bool = True
    run_test_sweep: bool = False
    rare_threshold_freq: int = 50

    # If true, evaluation label space uses train label mapping only. Labels in gold test not in mapping
    # are ignored by MultiLabelBinarizer, which avoids the appearance of test-label leakage.
    eval_labels_from_mapping_only: bool = True


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone encoder-aware LoRA hybrid routing for CPC classification.")

    p.add_argument("--dev-path", required=True)
    p.add_argument("--test-path", required=True)
    p.add_argument("--label-mapping", required=True)
    p.add_argument("--encoder-run-dir", required=True, help="Directory containing checkpoint-* subdirectories, or a checkpoint directory itself.")
    p.add_argument("--encoder-tokenizer-path", required=True)
    p.add_argument("--llm-base-path", required=True)
    p.add_argument("--llm-lora-path", default=None)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--cpc-definitions-path", default=None)
    p.add_argument("--rag-model-path", default=None)
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--offline", action="store_true", help="Set Hugging Face offline environment variables.")
    p.add_argument("--overwrite", action="store_true", help="Allow deletion of an existing output directory.")

    p.add_argument("--title-column", default="title")
    p.add_argument("--abstract-column", default="abstract")
    p.add_argument("--labels-column", default="labels")
    p.add_argument("--input-format", default="auto", choices=["auto", "csv", "tsv", "json", "jsonl"])

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--enc-batch-size", type=int, default=16)
    p.add_argument("--llm-batch-size", type=int, default=4)
    p.add_argument("--llm-max-new-tokens", type=int, default=128)
    p.add_argument("--llm-context-length", type=int, default=2048)
    p.add_argument("--min-labels", type=int, default=1)
    p.add_argument("--max-labels", type=int, default=7)

    p.add_argument("--routing-fractions", default="0.02,0.05,0.10,0.15,0.20")
    p.add_argument("--uncertainty-methods", default="maxprob,margin,entropy")
    p.add_argument("--merge-rules", default="audit_final", help="Comma-separated: audit_final,audit_plus_encoder_union,encoder_plus_llm_supported,encoder_plus_llm_overlap_only,encoder_replace_lowconf_with_supported")
    p.add_argument("--selection-metric", default="macro_f1", choices=["micro_f1", "macro_f1", "hier_f1", "routed_hier_delta", "routed_macro_delta"])

    p.add_argument("--encoder-topk-candidates", type=int, default=30)
    p.add_argument("--rag-topk-candidates", type=int, default=20)
    p.add_argument("--max-allowed-candidates-for-llm", type=int, default=45)
    p.add_argument("--max-encoder-topk-in-prompt", type=int, default=12)
    p.add_argument("--max-rag-topk-in-prompt", type=int, default=12)
    p.add_argument("--add-min-encoder-prob-grid", default="0.0")
    p.add_argument("--add-min-rag-score-grid", default="0.0")

    p.add_argument("--allow-free-llm-labels", action="store_true", help="Do not restrict LLM output to allowed candidates. Not recommended.")
    p.add_argument("--enable-regex-fallback", action="store_true", help="Allow CPC regex extraction when JSON parsing fails. Not recommended for final reported runs.")
    p.add_argument("--no-save-prompts", action="store_true")
    p.add_argument("--no-save-predictions", action="store_true")
    p.add_argument("--run-test-sweep", action="store_true", help="Exploratory only; do not use for final selection claims.")
    p.add_argument("--rare-threshold-freq", type=int, default=50)
    p.add_argument("--eval-labels-include-test-gold", action="store_true", help="Use mapping labels union test gold labels for evaluation bookkeeping. Default avoids this.")

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        dev_path=args.dev_path,
        test_path=args.test_path,
        label_mapping=args.label_mapping,
        encoder_run_dir=args.encoder_run_dir,
        encoder_tokenizer_path=args.encoder_tokenizer_path,
        llm_base_path=args.llm_base_path,
        llm_lora_path=args.llm_lora_path,
        cpc_definitions_path=args.cpc_definitions_path,
        rag_model_path=args.rag_model_path,
        out_dir=args.out_dir,
        no_rag=args.no_rag,
        offline=args.offline,
        overwrite=args.overwrite,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        labels_column=args.labels_column,
        input_format=args.input_format,
        seed=args.seed,
        max_length=args.max_length,
        enc_batch_size=args.enc_batch_size,
        llm_batch_size=args.llm_batch_size,
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_context_length=args.llm_context_length,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        routing_fractions=parse_float_list(args.routing_fractions),
        uncertainty_methods=parse_str_list(args.uncertainty_methods),
        merge_rules=parse_str_list(args.merge_rules),
        selection_metric=args.selection_metric,
        encoder_topk_candidates=args.encoder_topk_candidates,
        rag_topk_candidates=args.rag_topk_candidates,
        max_allowed_candidates_for_llm=args.max_allowed_candidates_for_llm,
        max_encoder_topk_in_prompt=args.max_encoder_topk_in_prompt,
        max_rag_topk_in_prompt=args.max_rag_topk_in_prompt,
        add_min_encoder_prob_grid=parse_float_list(args.add_min_encoder_prob_grid),
        add_min_rag_score_grid=parse_float_list(args.add_min_rag_score_grid),
        allow_llm_only_from_allowed_candidates=not args.allow_free_llm_labels,
        strict_no_regex_fallback=not args.enable_regex_fallback,
        save_prompts=not args.no_save_prompts,
        save_predictions=not args.no_save_predictions,
        run_test_sweep=args.run_test_sweep,
        rare_threshold_freq=args.rare_threshold_freq,
        eval_labels_from_mapping_only=not args.eval_labels_include_test_gold,
    )


# =============================================================================
# Utility
# =============================================================================

def configure_environment(cfg: Config) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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
        xn = normalize_cpc_label(x)
        if xn and xn not in seen:
            seen.add(xn)
            out.append(xn)
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
            return dedupe_preserve_order([p.strip() for p in s.split(";")])
        if "," in s:
            return dedupe_preserve_order([p.strip() for p in s.split(",")])
        return dedupe_preserve_order([s])
    return dedupe_preserve_order([raw])


def build_text(title: Any, abstract: Any) -> str:
    t = str(title or "").strip()
    a = str(abstract or "").strip()
    return f"{t}. {a}" if t and a else (t or a)


def truncate_text(s: Any, n: int = 220) -> str:
    text = " ".join(str(s or "").replace("\n", " ").split())
    return text if len(text) <= n else text[: max(0, n - 3)] + "..."


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def environment_report(cfg: Config) -> Dict[str, Any]:
    try:
        import transformers
    except Exception:
        transformers = None
    try:
        import peft
    except Exception:
        peft = None
    try:
        import sklearn
    except Exception:
        sklearn = None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": getattr(transformers, "__version__", None),
        "peft_version": getattr(peft, "__version__", None),
        "sklearn_version": getattr(sklearn, "__version__", None),
        "config": asdict(cfg),
    }


# =============================================================================
# Data and label mapping
# =============================================================================

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
    raise ValueError(f"Cannot infer input format from {path}; pass --input-format.")


def load_split(path: str, cfg: Config) -> Tuple[List[str], List[List[str]], List[str], List[str]]:
    dataset_type, delimiter = infer_dataset_format(path, cfg.input_format)
    kwargs: Dict[str, Any] = {"data_files": {"d": path}}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    ds = load_dataset(dataset_type, **kwargs)["d"]

    missing = [col for col in [cfg.title_column, cfg.abstract_column, cfg.labels_column] if col not in ds.column_names]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Available columns: {ds.column_names}")

    texts, labels, titles, abstracts = [], [], [], []
    for ex in ds:
        title = ex.get(cfg.title_column, "") or ""
        abstract = ex.get(cfg.abstract_column, "") or ""
        titles.append(str(title))
        abstracts.append(str(abstract))
        texts.append(build_text(title, abstract))
        labels.append(dedupe_preserve_order(parse_labels(ex.get(cfg.labels_column))))
    return texts, labels, titles, abstracts


def load_label_list(mapping_path: str) -> List[str]:
    p = Path(mapping_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing label mapping: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        labs = obj["labels"]
    elif isinstance(obj, dict) and "id2label" in obj:
        id2label = obj["id2label"]
        labs = [id2label[str(i)] for i in sorted(map(int, id2label.keys()))]
    elif isinstance(obj, dict) and "label2id" in obj:
        labs = [lab for lab, _ in sorted(obj["label2id"].items(), key=lambda kv: int(kv[1]))]
    elif isinstance(obj, dict) and all(str(k).isdigit() for k in obj.keys()):
        labs = [obj[str(i)] for i in sorted(map(int, obj.keys()))]
    elif isinstance(obj, list):
        labs = obj
    else:
        raise ValueError(f"Unrecognized label mapping format: {p}")

    labs = dedupe_preserve_order(labs)
    if not labs:
        raise ValueError(f"No labels parsed from {p}")
    return labs


# =============================================================================
# Encoder inference
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Encoder run/checkpoint directory not found: {p}")
    if (p / "config.json").is_file():
        return p
    ckpts = list(p.glob("checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* directories and no config.json found in {p}")

    def step(x: Path) -> int:
        try:
            return int(x.name.split("-")[-1])
        except Exception:
            return -1

    return sorted(ckpts, key=step)[-1]


def validate_model_dir(path: str, name: str) -> None:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"[{name}] Not a directory: {p}")
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"[{name}] Missing config.json in {p}")


def load_encoder(checkpoint_dir: Path, tokenizer_path: str, label_list: List[str]):
    validate_model_dir(str(checkpoint_dir), "encoder checkpoint")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=False, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir), local_files_only=False)
    model.config.id2label = {i: lab for i, lab in enumerate(label_list)}
    model.config.label2id = {lab: i for i, lab in enumerate(label_list)}

    out_dim = getattr(getattr(model, "classifier", None), "out_features", None)
    if out_dim is not None and out_dim != len(label_list):
        raise ValueError(f"Encoder classifier output size mismatch: {out_dim} vs {len(label_list)} labels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tok


def make_dataloader(texts: List[str], tokenizer, batch_size: int, max_length: int) -> DataLoader:
    ds = TextDataset(texts)

    def collate(batch_texts: List[str]):
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return enc

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def forward_probs(model, tokenizer, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    dl = make_dataloader(texts, tokenizer, batch_size, max_length)
    device = next(model.parameters()).device
    all_logits = []
    for enc in tqdm(dl, desc="Encoder forward", leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        all_logits.append(logits)
    return sigmoid(np.concatenate(all_logits, axis=0))


def decode_probs_to_labels(probs: np.ndarray, label_list: List[str], threshold: Optional[float], min_labels: int, max_labels: int) -> List[List[str]]:
    out = []
    for row in probs:
        pairs = sorted(zip(label_list, row.tolist()), key=lambda x: x[1], reverse=True)
        if threshold is None:
            chosen = [lab for lab, _ in pairs[:max_labels]]
        else:
            chosen = [lab for lab, p in pairs if p >= threshold]
            if len(chosen) < min_labels and pairs:
                chosen = [pairs[0][0]]
            if len(chosen) > max_labels:
                chosen = [lab for lab, _ in pairs[:max_labels]]
        out.append(dedupe_preserve_order(chosen))
    return out


def topk_candidates_from_probs(probs: np.ndarray, label_list: List[str], k: int) -> List[List[Tuple[str, float]]]:
    out = []
    for row in probs:
        pairs = sorted(zip(label_list, row.tolist()), key=lambda x: x[1], reverse=True)[:k]
        out.append([(lab, float(p)) for lab, p in pairs])
    return out


def micro_f1_from_sets(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = set(t), set(p)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def calibrate_threshold(dev_probs: np.ndarray, dev_gold: List[List[str]], label_list: List[str], cfg: Config) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in cfg.threshold_grid:
        pred = decode_probs_to_labels(dev_probs, label_list, float(t), cfg.min_labels, cfg.max_labels)
        f1 = micro_f1_from_sets(dev_gold, pred)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    print(f"[calibration] best dev micro-F1={best_f1:.6f} threshold={best_t:.4f}")
    return best_t


# =============================================================================
# RAG
# =============================================================================

def load_cpc_definitions(path: Optional[str], valid_labels: Set[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        print(f"[rag] CPC definitions not found: {p}. RAG disabled.")
        return {}

    out: Dict[str, str] = {}

    def add(code: Any, definition: Any) -> None:
        c = normalize_cpc_label(code)
        d = str(definition or "").strip()
        if c and d and c in valid_labels:
            out[c] = d

    try:
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        d = v.get("definition") or v.get("description") or v.get("text") or v.get("title")
                    else:
                        d = v
                    add(k, d)
            elif isinstance(obj, list):
                for row in obj:
                    if isinstance(row, dict):
                        code = row.get("code") or row.get("label") or row.get("cpc") or row.get("subclass")
                        definition = row.get("definition") or row.get("description") or row.get("text") or row.get("title")
                        add(code, definition)
        elif p.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    low = {str(k).lower(): v for k, v in row.items()}
                    code = low.get("code") or low.get("label") or low.get("cpc") or low.get("subclass")
                    definition = low.get("definition") or low.get("description") or low.get("text") or low.get("title")
                    add(code, definition)
        else:
            print(f"[rag] Unsupported CPC definitions format: {p.suffix}. RAG disabled.")
            return {}
    except Exception as e:
        print(f"[rag] Failed to load CPC definitions: {e}. RAG disabled.")
        return {}

    print(f"[rag] loaded {len(out)} CPC definitions")
    return out


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class RagRetriever:
    def __init__(self, model_path: Optional[str], cpc_definitions: Dict[str, str], batch_size: int = 64):
        self.enabled = False
        self.codes: List[str] = []
        self.definitions = cpc_definitions
        self.embeddings: Optional[torch.Tensor] = None
        self.batch_size = batch_size
        if not cpc_definitions or not model_path:
            return
        if not Path(model_path).is_dir():
            print(f"[rag] embedding model directory not found: {model_path}. RAG disabled.")
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False, use_fast=False)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.codes = sorted(cpc_definitions.keys())
        passages = [f"passage: {code}: {cpc_definitions[code]}" for code in self.codes]
        self.embeddings = self._embed_texts(passages, "RAG embed CPC definitions", batch_size)
        self.enabled = True
        print(f"[rag] retriever ready with {len(self.codes)} labels")

    @torch.no_grad()
    def _embed_texts(self, texts: List[str], desc: str, batch_size: int = 32) -> torch.Tensor:
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def retrieve(self, texts: List[str], topk: int) -> List[List[Tuple[str, float, str]]]:
        if not self.enabled or self.embeddings is None:
            return [[] for _ in texts]
        queries = ["query: " + " ".join(t.split()) for t in texts]
        q_emb = self._embed_texts(queries, "RAG embed patent queries", batch_size=16)
        sims = q_emb @ self.embeddings.T
        k = min(topk, sims.shape[1])
        top_scores, top_idx = torch.topk(sims, k=k, dim=1)
        rows = []
        for scores_row, idx_row in zip(top_scores.tolist(), top_idx.tolist()):
            cur = []
            for score, j in zip(scores_row, idx_row):
                code = self.codes[j]
                cur.append((code, float(score), self.definitions.get(code, "")))
            rows.append(cur)
        return rows


def merge_allowed_candidates(encoder_topk: List[Tuple[str, float]], rag_topk: List[Tuple[str, float, str]], max_allowed: int) -> List[str]:
    out, seen = [], set()
    enc_set = {lab for lab, _ in encoder_topk}
    rag_set = {lab for lab, _, _ in rag_topk}
    for lab, _ in encoder_topk:
        if lab in rag_set and lab not in seen:
            seen.add(lab)
            out.append(lab)
    for lab, _ in encoder_topk:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    for lab, _, _ in rag_topk:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out[:max_allowed]


# =============================================================================
# LLM audit
# =============================================================================

def load_llm(base_path: str, lora_path: Optional[str] = None):
    validate_model_dir(base_path, "LLM base")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=False, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=False, device_map="auto", torch_dtype=dtype).eval()
    if lora_path:
        if PeftModel is None:
            raise ImportError("peft is required for --llm-lora-path")
        if not Path(lora_path).is_dir():
            raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False).eval()
    return model, tok


def model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"


def format_encoder_predictions_block(pred_labels: List[str], prob_lookup: Dict[str, float]) -> str:
    if not pred_labels:
        return "- None"
    return "\n".join(f"- {lab} | encoder_prob={prob_lookup.get(lab, 0.0):.4f}" for lab in pred_labels)


def format_encoder_candidates_block(candidates: List[Tuple[str, float]], max_items: int) -> str:
    if not candidates:
        return "- None"
    return "\n".join(f"- {lab} | encoder_prob={p:.4f}" for lab, p in candidates[:max_items])


def format_rag_candidates_block(candidates: List[Tuple[str, float, str]], max_items: int) -> str:
    if not candidates:
        return "- None"
    lines = []
    for lab, score, definition in candidates[:max_items]:
        lines.append(f"- {lab} | rag_score={score:.4f} | definition={truncate_text(definition, 180)}")
    return "\n".join(lines)


def build_audit_prompt(tokenizer, cfg: Config, title: str, abstract: str, encoder_pred: List[str], encoder_candidates: List[Tuple[str, float]], rag_candidates: List[Tuple[str, float, str]], allowed_labels: List[str]) -> str:
    prob_lookup = {lab: p for lab, p in encoder_candidates}
    user_prompt = USER_PROMPT_AUDIT_TEMPLATE.format(
        title=truncate_text(title, 260),
        abstract=truncate_text(abstract, 1800),
        encoder_predictions_block=format_encoder_predictions_block(encoder_pred, prob_lookup),
        encoder_candidates_block=format_encoder_candidates_block(encoder_candidates, cfg.max_encoder_topk_in_prompt),
        rag_candidates_block=format_rag_candidates_block(rag_candidates, cfg.max_rag_topk_in_prompt),
        allowed_labels_block=", ".join(allowed_labels) if allowed_labels else "- None",
        max_labels=cfg.max_labels,
    )
    return build_chat_prompt(tokenizer, SYSTEM_PROMPT_AUDIT, user_prompt)


def filter_labels(labels: Iterable[Any], allowed_set: Optional[Set[str]], max_labels: int) -> List[str]:
    out, seen = [], set()
    for x in labels:
        c = normalize_cpc_label(x)
        if not c:
            continue
        if allowed_set is not None and c not in allowed_set:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_labels:
            break
    return out


def parse_llm_labels(text: str, allowed_set: Optional[Set[str]], cfg: Config) -> Tuple[List[str], str]:
    raw = (text or "").strip()
    if not raw:
        return [], "empty"

    def from_obj(obj: Any) -> List[str]:
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return filter_labels(obj["labels"], allowed_set, cfg.max_labels)
        if isinstance(obj, list):
            return filter_labels(obj, allowed_set, cfg.max_labels)
        return []

    try:
        labs = from_obj(json.loads(raw))
        if labs:
            return labs, "strict_json"
    except Exception:
        pass

    match_obj = _JSON_OBJ_RE.search(raw)
    if match_obj:
        try:
            labs = from_obj(json.loads(match_obj.group(0)))
            if labs:
                return labs, "embedded_json"
        except Exception:
            pass

    match_list = _JSON_LIST_RE.search(raw)
    if match_list:
        try:
            labs = from_obj(json.loads(match_list.group(0)))
            if labs:
                return labs, "embedded_json_list"
        except Exception:
            pass

    if not cfg.strict_no_regex_fallback:
        labs = filter_labels(_CPC_RE.findall(raw.upper()), allowed_set, cfg.max_labels)
        if labs:
            return labs, "regex_cpc_tokens"

    return [], "parse_failed"


@torch.no_grad()
def generate_llm_batch(model, tokenizer, prompts: List[str], cfg: Config) -> List[str]:
    device = model_device(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.llm_context_length)
    # Track prompt truncation in attention lengths indirectly in saved diagnostics if needed.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.llm_max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = []
    for i in range(len(prompts)):
        prompt_len = int(inputs["attention_mask"][i].sum().item())
        gen_ids = outputs[i][prompt_len:]
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return decoded


def run_llm_audit_on_indices(model, tokenizer, cfg: Config, titles: List[str], abstracts: List[str], encoder_preds: List[List[str]], encoder_topk: List[List[Tuple[str, float]]], rag_topk: List[List[Tuple[str, float, str]]], allowed_candidates: List[List[str]], route_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for start in tqdm(range(0, len(route_indices), cfg.llm_batch_size), desc="LLM audit", leave=False):
        batch_idx = route_indices[start:start + cfg.llm_batch_size]
        prompts, meta = [], []
        for i in batch_idx:
            prompt = build_audit_prompt(tokenizer, cfg, titles[i], abstracts[i], encoder_preds[i], encoder_topk[i], rag_topk[i], allowed_candidates[i])
            prompts.append(prompt)
            meta.append({"idx": i, "allowed": allowed_candidates[i]})
        raw_outputs = generate_llm_batch(model, tokenizer, prompts, cfg)
        for prompt, raw, record in zip(prompts, raw_outputs, meta):
            idx = record["idx"]
            allowed_set = set(record["allowed"]) if cfg.allow_llm_only_from_allowed_candidates else None
            labels, parse_mode = parse_llm_labels(raw, allowed_set, cfg)
            used_fallback = False
            if not labels:
                labels = list(encoder_preds[idx])
                used_fallback = True
            results[idx] = {
                "pred": labels[:cfg.max_labels],
                "raw_generation": raw,
                "prompt": prompt if cfg.save_prompts else None,
                "parse_meta": {
                    "parse_mode": parse_mode,
                    "used_fallback_to_encoder": used_fallback,
                    "empty_after_processing": len(labels) == 0,
                    "allowed_set_size": len(record["allowed"]),
                },
            }
    return results


# =============================================================================
# Routing, merging, metrics
# =============================================================================

def row_entropy(prob_row: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(prob_row.astype(np.float64), eps, 1.0 - eps)
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())


def compute_uncertainty_scores(probs: np.ndarray, method: str) -> np.ndarray:
    if method == "maxprob":
        return 1.0 - probs.max(axis=1)
    if method == "margin":
        if probs.shape[1] < 2:
            return np.ones(probs.shape[0], dtype=np.float64)
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        margin = np.max(top2, axis=1) - np.min(top2, axis=1)
        return 1.0 - margin
    if method == "entropy":
        return np.array([row_entropy(row) for row in probs], dtype=np.float64)
    raise ValueError(f"Unknown uncertainty method: {method}")


def merge_predictions(enc_pred: List[str], llm_pred: List[str], merge_rule: str, cfg: Config, encoder_candidates: Optional[List[Tuple[str, float]]] = None, rag_candidates: Optional[List[Tuple[str, float, str]]] = None, add_min_encoder_prob: float = 0.0, add_min_rag_score: float = 0.0, encoder_threshold: Optional[float] = None) -> List[str]:
    enc_pred = dedupe_preserve_order(enc_pred)
    llm_pred = dedupe_preserve_order(llm_pred)
    enc_probs = {normalize_cpc_label(lab): float(p) for lab, p in (encoder_candidates or [])}
    rag_scores = {normalize_cpc_label(lab): float(s) for lab, s, _ in (rag_candidates or [])}

    def supported(labels: List[str]) -> List[str]:
        out = []
        for lab in labels:
            enc_ok = enc_probs.get(lab, 0.0) >= add_min_encoder_prob
            rag_ok = rag_scores.get(lab, -1.0) >= add_min_rag_score
            if enc_ok or rag_ok:
                out.append(lab)
        return dedupe_preserve_order(out)

    if merge_rule == "audit_final":
        merged = llm_pred if llm_pred else enc_pred
    elif merge_rule == "audit_plus_encoder_union":
        merged = dedupe_preserve_order(enc_pred + llm_pred)
    elif merge_rule == "encoder_plus_llm_supported":
        merged = dedupe_preserve_order(enc_pred + supported(llm_pred))
    elif merge_rule == "encoder_plus_llm_overlap_only":
        overlap = [lab for lab in llm_pred if lab in enc_probs and lab in rag_scores]
        merged = dedupe_preserve_order(enc_pred + overlap)
    elif merge_rule == "encoder_replace_lowconf_with_supported":
        keep_floor = encoder_threshold if encoder_threshold is not None else add_min_encoder_prob
        strong_encoder = [lab for lab in enc_pred if enc_probs.get(lab, 1.0) >= keep_floor]
        if not strong_encoder:
            strong_encoder = enc_pred[:1]
        merged = dedupe_preserve_order(strong_encoder + supported(llm_pred))
    else:
        raise ValueError(f"Unknown merge rule: {merge_rule}")

    if not merged and enc_pred:
        merged = enc_pred[:1]
    return merged[:cfg.max_labels]


def build_hier_tags(labels: List[str]) -> List[str]:
    tags = []
    for lab in labels:
        c = normalize_cpc_label(lab)
        if len(c) >= 1 and c[0].isalpha():
            tags.append(f"SECTION_{c[0]}")
        if len(c) >= 3 and c[0].isalpha() and c[1:3].isdigit():
            tags.append(f"CLASS_{c[:3]}")
        if len(c) >= 4:
            tags.append(f"SUBCLASS_{c[:4]}")
    return sorted(set(tags))


def hierarchical_micro_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = set(build_hier_tags(t)), set(build_hier_tags(p))
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def flat_acc_at_1(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    total = correct = 0
    for t, p in zip(y_true, y_pred):
        if p:
            total += 1
            correct += int(p[0] in set(t))
    return correct / total if total else 0.0


def label_count_diagnostics(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    return {
        "avg_gold_labels_per_patent": float(np.mean([len(x) for x in gold])) if gold else 0.0,
        "avg_pred_labels_per_patent": float(np.mean([len(x) for x in pred])) if pred else 0.0,
        "empty_prediction_rate": float(np.mean([1.0 if not x else 0.0 for x in pred])) if pred else 0.0,
    }


def evaluate_predictions(gold: List[List[str]], pred: List[List[str]], eval_label_space: List[str]) -> Dict[str, Any]:
    mlb = MultiLabelBinarizer(classes=eval_label_space)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    h_p, h_r, h_f1 = hierarchical_micro_metrics(gold, pred)
    return {
        "flat_multilabel_subclass": {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "acc_at_1": float(flat_acc_at_1(gold, pred)),
        },
        "hierarchical_micro_tags": {"precision": float(h_p), "recall": float(h_r), "f1": float(h_f1)},
        "label_count_diagnostics": label_count_diagnostics(gold, pred),
    }


def label_frequencies(labels: List[List[str]]) -> Counter:
    c = Counter()
    for labs in labels:
        for lab in labs:
            lab = normalize_cpc_label(lab)
            if lab:
                c[lab] += 1
    return c


def filter_by_label_set(labels: List[List[str]], allowed: Set[str]) -> List[List[str]]:
    return [dedupe_preserve_order([lab for lab in labs if normalize_cpc_label(lab) in allowed]) for labs in labels]


def evaluate_subset(gold: List[List[str]], pred: List[List[str]], label_space: List[str]) -> Dict[str, Any]:
    if not label_space:
        return {"micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "n_labels": 0}
    mlb = MultiLabelBinarizer(classes=label_space)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1), "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1), "n_labels": len(label_space)}


def evaluate_rare_analysis(gold: List[List[str]], enc_pred: List[List[str]], hybrid_pred: List[List[str]], rare_threshold: int) -> Dict[str, Any]:
    freq = label_frequencies(gold)
    rare = {lab for lab, n in freq.items() if n < rare_threshold}
    frequent = {lab for lab, n in freq.items() if n >= rare_threshold}
    gold_rare, enc_rare, hyb_rare = filter_by_label_set(gold, rare), filter_by_label_set(enc_pred, rare), filter_by_label_set(hybrid_pred, rare)
    gold_freq, enc_freq, hyb_freq = filter_by_label_set(gold, frequent), filter_by_label_set(enc_pred, frequent), filter_by_label_set(hybrid_pred, frequent)

    correct_rare_additions = incorrect_rare_additions = rare_recoveries = 0
    for g, e, h in zip(gold, enc_pred, hybrid_pred):
        gs, es, hs = set(g), set(e), set(h)
        added_rare = (hs - es) & rare
        correct_rare_additions += len(added_rare & gs)
        incorrect_rare_additions += len(added_rare - gs)
        rare_recoveries += len(((gs & rare) - es) & hs)

    return {
        "rare_threshold_freq": rare_threshold,
        "n_rare_labels": len(rare),
        "n_frequent_labels": len(frequent),
        "encoder_rare_only_metrics": evaluate_subset(gold_rare, enc_rare, sorted(rare)),
        "hybrid_rare_only_metrics": evaluate_subset(gold_rare, hyb_rare, sorted(rare)),
        "encoder_frequent_only_metrics": evaluate_subset(gold_freq, enc_freq, sorted(frequent)),
        "hybrid_frequent_only_metrics": evaluate_subset(gold_freq, hyb_freq, sorted(frequent)),
        "rare_utility": {
            "correct_rare_additions": correct_rare_additions,
            "incorrect_rare_additions": incorrect_rare_additions,
            "net_rare_additions": correct_rare_additions - incorrect_rare_additions,
            "rare_recoveries_from_encoder_misses": rare_recoveries,
        },
    }


def evaluate_routed_subset(gold: List[List[str]], enc_pred: List[List[str]], hybrid_pred: List[List[str]], route_indices: List[int], eval_label_space: List[str]) -> Dict[str, Any]:
    idxs = sorted(route_indices)
    if not idxs:
        return {}
    g = [gold[i] for i in idxs]
    e = [enc_pred[i] for i in idxs]
    h = [hybrid_pred[i] for i in idxs]
    em = evaluate_predictions(g, e, eval_label_space)
    hm = evaluate_predictions(g, h, eval_label_space)
    return {
        "n_routed": len(idxs),
        "encoder": em,
        "hybrid": hm,
        "routed_subset_delta": {
            "micro_f1": hm["flat_multilabel_subclass"]["micro_f1"] - em["flat_multilabel_subclass"]["micro_f1"],
            "macro_f1": hm["flat_multilabel_subclass"]["macro_f1"] - em["flat_multilabel_subclass"]["macro_f1"],
            "hier_f1": hm["hierarchical_micro_tags"]["f1"] - em["hierarchical_micro_tags"]["f1"],
        },
    }


def parse_diagnostics(llm_outputs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    modes = Counter()
    fallback = 0
    for rec in llm_outputs.values():
        meta = rec.get("parse_meta", {})
        modes[meta.get("parse_mode", "unknown")] += 1
        fallback += int(bool(meta.get("used_fallback_to_encoder")))
    n = len(llm_outputs)
    return {"n_routed": n, "parse_mode_counts": dict(modes), "fallback_to_encoder": fallback, "fallback_rate": fallback / n if n else 0.0}


# =============================================================================
# Hybrid evaluation
# =============================================================================

def evaluate_hybrid_for_split(split_name: str, cfg: Config, probs: np.ndarray, gold: List[List[str]], titles: List[str], abstracts: List[str], enc_preds: List[List[str]], enc_topk: List[List[Tuple[str, float]]], rag_topk: List[List[Tuple[str, float, str]]], allowed_candidates: List[List[str]], eval_label_space: List[str], llm_model, llm_tok, uncertainty_method: str, routing_fraction: float, merge_rule: str, add_min_encoder_prob: float, add_min_rag_score: float, encoder_threshold: Optional[float], out_dir: Path) -> Dict[str, Any]:
    scores = compute_uncertainty_scores(probs, uncertainty_method)
    n_total = probs.shape[0]
    n_route = max(1, int(round(routing_fraction * n_total)))
    route_indices = np.argsort(-scores)[:n_route].tolist()
    route_set = set(route_indices)

    print(f"[{split_name}] method={uncertainty_method} frac={routing_fraction:.2f} routed={n_route}/{n_total} merge={merge_rule}")

    llm_outputs = run_llm_audit_on_indices(llm_model, llm_tok, cfg, titles, abstracts, enc_preds, enc_topk, rag_topk, allowed_candidates, route_indices)
    hybrid_preds: List[List[str]] = []
    for i in range(n_total):
        if i not in route_set:
            hybrid_preds.append(enc_preds[i])
        else:
            hybrid_preds.append(merge_predictions(enc_preds[i], llm_outputs[i]["pred"], merge_rule, cfg, enc_topk[i], rag_topk[i], add_min_encoder_prob, add_min_rag_score, encoder_threshold))

    metrics = evaluate_predictions(gold, hybrid_preds, eval_label_space)
    encoder_metrics = evaluate_predictions(gold, enc_preds, eval_label_space)
    rare_analysis = evaluate_rare_analysis(gold, enc_preds, hybrid_preds, cfg.rare_threshold_freq)
    routed_subset_metrics = evaluate_routed_subset(gold, enc_preds, hybrid_preds, route_indices, eval_label_space)
    diag = parse_diagnostics(llm_outputs)

    result = {
        "split": split_name,
        "uncertainty_method": uncertainty_method,
        "routing_fraction": routing_fraction,
        "num_routed": n_route,
        "merge_rule": merge_rule,
        "add_min_encoder_prob": add_min_encoder_prob,
        "add_min_rag_score": add_min_rag_score,
        "metrics": metrics,
        "encoder_metrics": encoder_metrics,
        "full_dataset_delta": {
            "micro_f1": metrics["flat_multilabel_subclass"]["micro_f1"] - encoder_metrics["flat_multilabel_subclass"]["micro_f1"],
            "macro_f1": metrics["flat_multilabel_subclass"]["macro_f1"] - encoder_metrics["flat_multilabel_subclass"]["macro_f1"],
            "hier_f1": metrics["hierarchical_micro_tags"]["f1"] - encoder_metrics["hierarchical_micro_tags"]["f1"],
        },
        "routed_subset_metrics": routed_subset_metrics,
        "rare_analysis": rare_analysis,
        "llm_parse_diagnostics": diag,
        "candidate_policy": {
            "encoder_topk_candidates": cfg.encoder_topk_candidates,
            "rag_topk_candidates": cfg.rag_topk_candidates,
            "max_allowed_candidates_for_llm": cfg.max_allowed_candidates_for_llm,
            "allow_llm_only_from_allowed_candidates": cfg.allow_llm_only_from_allowed_candidates,
            "strict_no_regex_fallback": cfg.strict_no_regex_fallback,
        },
    }

    tag = f"{split_name}_{uncertainty_method}_frac{int(round(100 * routing_fraction))}_{merge_rule}_ep{int(100 * add_min_encoder_prob)}_rs{int(100 * add_min_rag_score)}"
    save_json(out_dir / f"hybrid_{tag}.json", result)

    if cfg.save_predictions:
        with (out_dir / f"predictions_{tag}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(n_total):
                rec = {
                    "idx": i,
                    "title": titles[i],
                    "abstract": abstracts[i],
                    "gold_labels": gold[i],
                    "encoder_pred": enc_preds[i],
                    "encoder_topk": enc_topk[i],
                    "rag_topk": rag_topk[i],
                    "allowed_candidates": allowed_candidates[i],
                    "hybrid_pred": hybrid_preds[i],
                    "routed_to_llm": i in route_set,
                    "uncertainty_score": float(scores[i]),
                }
                if i in route_set:
                    llm_rec = dict(llm_outputs[i])
                    if not cfg.save_prompts:
                        llm_rec.pop("prompt", None)
                    rec["llm"] = llm_rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    flat = metrics["flat_multilabel_subclass"]
    hier = metrics["hierarchical_micro_tags"]
    rd = routed_subset_metrics.get("routed_subset_delta", {})
    print(f"  full micro={flat['micro_f1']:.4f} macro={flat['macro_f1']:.4f} h={hier['f1']:.4f} | routed deltas={rd}")
    return result


def selection_score(result: Dict[str, Any], metric: str) -> Tuple[float, float]:
    flat = result["metrics"]["flat_multilabel_subclass"]
    hier = result["metrics"]["hierarchical_micro_tags"]
    routed_delta = result.get("routed_subset_metrics", {}).get("routed_subset_delta", {})
    if metric == "micro_f1":
        return flat["micro_f1"], flat["macro_f1"]
    if metric == "macro_f1":
        return flat["macro_f1"], flat["micro_f1"]
    if metric == "hier_f1":
        return hier["f1"], flat["macro_f1"]
    if metric == "routed_hier_delta":
        return routed_delta.get("hier_f1", -math.inf), hier["f1"]
    if metric == "routed_macro_delta":
        return routed_delta.get("macro_f1", -math.inf), flat["macro_f1"]
    raise ValueError(f"Unknown selection metric: {metric}")


def select_on_dev(cfg: Config, dev_payload: Dict[str, Any], eval_label_space: List[str], llm_model, llm_tok, encoder_threshold: Optional[float], out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results = []
    best = None
    best_score = (-math.inf, -math.inf)
    for merge_rule in cfg.merge_rules:
        prob_grid = cfg.add_min_encoder_prob_grid if merge_rule in {"encoder_plus_llm_supported", "encoder_replace_lowconf_with_supported"} else [0.0]
        rag_grid = cfg.add_min_rag_score_grid if merge_rule in {"encoder_plus_llm_supported", "encoder_replace_lowconf_with_supported"} else [0.0]
        for ep in prob_grid:
            for rs in rag_grid:
                for method in cfg.uncertainty_methods:
                    for frac in cfg.routing_fractions:
                        res = evaluate_hybrid_for_split(
                            "dev",
                            cfg,
                            dev_payload["probs"],
                            dev_payload["gold"],
                            dev_payload["titles"],
                            dev_payload["abstracts"],
                            dev_payload["enc_preds"],
                            dev_payload["enc_topk"],
                            dev_payload["rag_topk"],
                            dev_payload["allowed_candidates"],
                            eval_label_space,
                            llm_model,
                            llm_tok,
                            method,
                            frac,
                            merge_rule,
                            ep,
                            rs,
                            encoder_threshold,
                            out_dir,
                        )
                        results.append(res)
                        score = selection_score(res, cfg.selection_metric)
                        if score > best_score:
                            best_score = score
                            best = res
    if best is None:
        raise RuntimeError("No DEV hybrid configuration evaluated.")
    save_json(out_dir / "dev_hybrid_selection_summary.json", {"selection_metric": cfg.selection_metric, "best_score": best_score, "best_result": best, "all_results": results})
    return best, results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    configure_environment(cfg)
    set_all_seeds(cfg.seed)

    out_dir = Path(cfg.out_dir)
    ensure_out_dir(out_dir, cfg.overwrite)
    save_json(out_dir / "run_environment.json", environment_report(cfg))

    print("[data] loading label mapping and splits")
    label_list = load_label_list(cfg.label_mapping)
    label_set = set(label_list)
    dev_texts, dev_gold, dev_titles, dev_abstracts = load_split(cfg.dev_path, cfg)
    test_texts, test_gold, test_titles, test_abstracts = load_split(cfg.test_path, cfg)

    if cfg.eval_labels_from_mapping_only:
        eval_label_space = list(label_list)
        unseen_test_labels = sorted({lab for labs in test_gold for lab in labs if lab not in label_set})
    else:
        eval_label_space = sorted(set(label_list) | {lab for labs in test_gold for lab in labs})
        unseen_test_labels = []

    save_json(out_dir / "data_summary.json", {
        "n_dev": len(dev_texts),
        "n_test": len(test_texts),
        "n_label_mapping": len(label_list),
        "n_eval_label_space": len(eval_label_space),
        "unseen_test_labels_ignored": unseen_test_labels,
        "avg_dev_gold_labels": float(np.mean([len(x) for x in dev_gold])),
        "avg_test_gold_labels": float(np.mean([len(x) for x in test_gold])),
    })

    print(f"[data] dev={len(dev_texts)} test={len(test_texts)} labels={len(label_list)}")

    print("[encoder] loading")
    ckpt_dir = resolve_checkpoint(cfg.encoder_run_dir)
    enc_model, enc_tok = load_encoder(ckpt_dir, cfg.encoder_tokenizer_path, label_list)

    print("[encoder] forward DEV")
    dev_probs = forward_probs(enc_model, enc_tok, dev_texts, cfg.enc_batch_size, cfg.max_length)
    print("[encoder] forward TEST")
    test_probs = forward_probs(enc_model, enc_tok, test_texts, cfg.enc_batch_size, cfg.max_length)

    threshold = calibrate_threshold(dev_probs, dev_gold, label_list, cfg)
    dev_enc_preds = decode_probs_to_labels(dev_probs, label_list, threshold, cfg.min_labels, cfg.max_labels)
    test_enc_preds = decode_probs_to_labels(test_probs, label_list, threshold, cfg.min_labels, cfg.max_labels)
    dev_enc_topk = topk_candidates_from_probs(dev_probs, label_list, cfg.encoder_topk_candidates)
    test_enc_topk = topk_candidates_from_probs(test_probs, label_list, cfg.encoder_topk_candidates)

    print("[rag] preparing candidates")
    if cfg.no_rag:
        cpc_defs = {}
        dev_rag_topk = [[] for _ in dev_texts]
        test_rag_topk = [[] for _ in test_texts]
    else:
        cpc_defs = load_cpc_definitions(cfg.cpc_definitions_path, label_set)
        retriever = RagRetriever(cfg.rag_model_path, cpc_defs)
        if retriever.enabled:
            dev_rag_topk = retriever.retrieve(dev_texts, cfg.rag_topk_candidates)
            test_rag_topk = retriever.retrieve(test_texts, cfg.rag_topk_candidates)
        else:
            dev_rag_topk = [[] for _ in dev_texts]
            test_rag_topk = [[] for _ in test_texts]

    dev_allowed = [merge_allowed_candidates(e, r, cfg.max_allowed_candidates_for_llm) for e, r in zip(dev_enc_topk, dev_rag_topk)]
    test_allowed = [merge_allowed_candidates(e, r, cfg.max_allowed_candidates_for_llm) for e, r in zip(test_enc_topk, test_rag_topk)]

    encoder_test_metrics = evaluate_predictions(test_gold, test_enc_preds, eval_label_space)
    save_json(out_dir / "encoder_baseline_metrics.json", {
        "encoder_checkpoint": str(ckpt_dir),
        "encoder_threshold": threshold,
        "metrics": encoder_test_metrics,
        "rare_analysis": evaluate_rare_analysis(test_gold, test_enc_preds, test_enc_preds, cfg.rare_threshold_freq),
    })

    print("[llm] loading audit model")
    llm_model, llm_tok = load_llm(cfg.llm_base_path, cfg.llm_lora_path)

    dev_payload = {
        "probs": dev_probs,
        "gold": dev_gold,
        "titles": dev_titles,
        "abstracts": dev_abstracts,
        "enc_preds": dev_enc_preds,
        "enc_topk": dev_enc_topk,
        "rag_topk": dev_rag_topk,
        "allowed_candidates": dev_allowed,
    }
    test_payload = {
        "probs": test_probs,
        "gold": test_gold,
        "titles": test_titles,
        "abstracts": test_abstracts,
        "enc_preds": test_enc_preds,
        "enc_topk": test_enc_topk,
        "rag_topk": test_rag_topk,
        "allowed_candidates": test_allowed,
    }

    best_dev, all_dev = select_on_dev(cfg, dev_payload, eval_label_space, llm_model, llm_tok, threshold, out_dir)
    selected = {
        "uncertainty_method": best_dev["uncertainty_method"],
        "routing_fraction": best_dev["routing_fraction"],
        "merge_rule": best_dev["merge_rule"],
        "add_min_encoder_prob": best_dev.get("add_min_encoder_prob", 0.0),
        "add_min_rag_score": best_dev.get("add_min_rag_score", 0.0),
        "selection_metric": cfg.selection_metric,
    }
    save_json(out_dir / "selected_dev_config.json", selected)

    print("[test] evaluating selected DEV configuration once")
    test_result = evaluate_hybrid_for_split(
        "test_selected",
        cfg,
        test_payload["probs"],
        test_payload["gold"],
        test_payload["titles"],
        test_payload["abstracts"],
        test_payload["enc_preds"],
        test_payload["enc_topk"],
        test_payload["rag_topk"],
        test_payload["allowed_candidates"],
        eval_label_space,
        llm_model,
        llm_tok,
        selected["uncertainty_method"],
        selected["routing_fraction"],
        selected["merge_rule"],
        selected["add_min_encoder_prob"],
        selected["add_min_rag_score"],
        threshold,
        out_dir,
    )

    final_summary = {
        "encoder_threshold": threshold,
        "selected_on_dev": selected,
        "encoder_test_metrics": encoder_test_metrics,
        "test_result": test_result,
    }
    save_json(out_dir / "hybrid_selected_test_result.json", final_summary)

    if cfg.run_test_sweep:
        print("[warning] running exploratory TEST sweep. Do not use for final model selection claims.")
        test_sweep = []
        for merge_rule in cfg.merge_rules:
            prob_grid = cfg.add_min_encoder_prob_grid if merge_rule in {"encoder_plus_llm_supported", "encoder_replace_lowconf_with_supported"} else [0.0]
            rag_grid = cfg.add_min_rag_score_grid if merge_rule in {"encoder_plus_llm_supported", "encoder_replace_lowconf_with_supported"} else [0.0]
            for ep in prob_grid:
                for rs in rag_grid:
                    for method in cfg.uncertainty_methods:
                        for frac in cfg.routing_fractions:
                            test_sweep.append(evaluate_hybrid_for_split(
                                "test_exploratory",
                                cfg,
                                test_payload["probs"],
                                test_payload["gold"],
                                test_payload["titles"],
                                test_payload["abstracts"],
                                test_payload["enc_preds"],
                                test_payload["enc_topk"],
                                test_payload["rag_topk"],
                                test_payload["allowed_candidates"],
                                eval_label_space,
                                llm_model,
                                llm_tok,
                                method,
                                frac,
                                merge_rule,
                                ep,
                                rs,
                                threshold,
                                out_dir,
                            ))
        save_json(out_dir / "test_exploratory_sweep_summary.json", test_sweep)

    del llm_model, llm_tok, enc_model, enc_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nSaved outputs to:")
    print(out_dir)


if __name__ == "__main__":
    main()
