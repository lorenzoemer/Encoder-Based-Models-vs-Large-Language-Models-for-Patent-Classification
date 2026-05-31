#!/usr/bin/env python3
"""

What this script does:
  1. Loads train/dev patent data from TSV, CSV, JSON, or JSONL.
  2. Formats each example as supervised instruction tuning data.
  3. Fine-tunes an open-weight causal LM with LoRA adapters.
  4. Selects the best checkpoint by validation loss with early stopping.
  5. Saves the final LoRA adapter and tokenizer.
  6. Evaluates the adapter on the dev split by deterministic generation.
  7. Exports metrics, predictions, learning curves, run configuration, and environment metadata.

Expected input columns by default:
  title, abstract, labels

Accepted label formats:
  - JSON list: ["G06F", "H04L"]
  - Python-style list: ['G06F', 'H04L']
  - Semicolon-separated string: G06F;H04L
  - Comma-separated string: G06F,H04L

Example:
  python qwen_lora_cpc_train_publishable.py \
    --model-path models/Qwen2.5-7B-Instruct \
    --train-path data/train.tsv \
    --dev-path data/dev.tsv \
    --output-dir outputs/qwen2_5_7b_cpc_lora \
    --input-format tsv \
    --offline \
    --overwrite

"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import inspect
import json
import os
import platform
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

try:
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover
    EmissionsTracker = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    model_path: str
    train_path: str
    dev_path: str
    output_dir: str

    input_format: str = "auto"  # auto, tsv, csv, json, jsonl
    title_column: str = "title"
    abstract_column: str = "abstract"
    labels_column: str = "labels"

    offline: bool = False
    overwrite: bool = False
    seed: int = 1

    max_seq_length: int = 2048
    max_new_tokens: int = 256
    generation_batch_size: int = 1

    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    adapter_subdir: str = "best_lora_adapter"
    use_codecarbon: bool = True
    codecarbon_project_name: str = "qwen_lora_cpc_classification"
    codecarbon_measure_power_secs: int = 10

    max_eval_examples: Optional[int] = None
    max_train_examples: Optional[int] = None
    max_dev_examples: Optional[int] = None

    # If supplied, evaluation label space is fixed to this file instead of train+dev labels.
    label_space_path: Optional[str] = None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a Qwen-style causal LM with LoRA for CPC multi-label classification.")

    p.add_argument("--model-path", required=True, help="Local path or Hugging Face model id.")
    p.add_argument("--train-path", required=True)
    p.add_argument("--dev-path", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--input-format", default="auto", choices=["auto", "tsv", "csv", "json", "jsonl"])
    p.add_argument("--title-column", default="title")
    p.add_argument("--abstract-column", default="abstract")
    p.add_argument("--labels-column", default="labels")

    p.add_argument("--offline", action="store_true", help="Use local files only and set HF offline environment variables.")
    p.add_argument("--overwrite", action="store_true", help="Delete output directory if it already exists.")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--generation-batch-size", type=int, default=1)

    p.add_argument("--num-train-epochs", type=float, default=10.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--early-stopping-threshold", type=float, default=0.001)

    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    p.add_argument("--adapter-subdir", default="best_lora_adapter")
    p.add_argument("--no-codecarbon", action="store_true")
    p.add_argument("--codecarbon-project-name", default="qwen_lora_cpc_classification")
    p.add_argument("--codecarbon-measure-power-secs", type=int, default=10)

    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-dev-examples", type=int, default=None)
    p.add_argument("--label-space-path", default=None, help="Optional JSON list of CPC labels for evaluation.")

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        model_path=args.model_path,
        train_path=args.train_path,
        dev_path=args.dev_path,
        output_dir=args.output_dir,
        input_format=args.input_format,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        labels_column=args.labels_column,
        offline=args.offline,
        overwrite=args.overwrite,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        adapter_subdir=args.adapter_subdir,
        use_codecarbon=not args.no_codecarbon,
        codecarbon_project_name=args.codecarbon_project_name,
        codecarbon_measure_power_secs=args.codecarbon_measure_power_secs,
        max_eval_examples=args.max_eval_examples,
        max_train_examples=args.max_train_examples,
        max_dev_examples=args.max_dev_examples,
        label_space_path=args.label_space_path,
    )


# =============================================================================
# General utilities
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
        import peft
    except Exception:
        peft = None
    try:
        import trl
    except Exception:
        trl = None
    try:
        import sklearn
    except Exception:
        sklearn = None
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
        "peft_version": getattr(peft, "__version__", None),
        "trl_version": getattr(trl, "__version__", None),
        "sklearn_version": getattr(sklearn, "__version__", None),
        "codecarbon_version": getattr(codecarbon, "__version__", None),
        "config": asdict(cfg),
    }


def dtype_for_training() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


# =============================================================================
# Data loading and label parsing
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


def load_raw_datasets(cfg: Config) -> DatasetDict:
    dataset_type, delimiter = infer_dataset_format(cfg.train_path, cfg.input_format)
    dev_type, dev_delimiter = infer_dataset_format(cfg.dev_path, cfg.input_format)
    if dataset_type != dev_type or delimiter != dev_delimiter:
        raise ValueError("Train and dev files must use the same format for this script.")

    kwargs: Dict[str, Any] = {"data_files": {"train": cfg.train_path, "validation": cfg.dev_path}}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter

    raw = load_dataset(dataset_type, **kwargs)
    required = [cfg.title_column, cfg.abstract_column, cfg.labels_column]
    for split in ["train", "validation"]:
        missing = [c for c in required if c not in raw[split].column_names]
        if missing:
            raise ValueError(f"Missing columns in {split}: {missing}. Available: {raw[split].column_names}")

    if cfg.max_train_examples is not None:
        raw["train"] = raw["train"].select(range(min(cfg.max_train_examples, len(raw["train"]))))
    if cfg.max_dev_examples is not None:
        raw["validation"] = raw["validation"].select(range(min(cfg.max_dev_examples, len(raw["validation"]))))

    return raw


def normalize_code(code: Any) -> str:
    c = str(code or "").strip().upper()
    if not c:
        return ""
    c = c.split("/")[0]
    if len(c) >= 4 and c[0].isalpha() and c[1:3].isdigit():
        return c[:4]
    return c[:4]


def normalize_labels(labels: Iterable[Any]) -> List[str]:
    out: Set[str] = set()
    for x in labels:
        c = normalize_code(x)
        if c:
            out.add(c)
    return sorted(out)


def parse_labels(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return normalize_labels(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, list):
                    return normalize_labels(obj)
            except Exception:
                pass
        if ";" in s:
            return normalize_labels([x.strip() for x in s.split(";")])
        if "," in s:
            return normalize_labels([x.strip() for x in s.split(",")])
        return normalize_labels([s])
    return normalize_labels([raw])


def build_input_text(example: Dict[str, Any], cfg: Config) -> str:
    title = str(example.get(cfg.title_column, "") or "").strip()
    abstract = str(example.get(cfg.abstract_column, "") or "").strip()
    return f"{title}. {abstract}".strip() if title and abstract else (title or abstract)


def collect_label_space(raw: DatasetDict, cfg: Config) -> List[str]:
    if cfg.label_space_path:
        p = Path(cfg.label_space_path)
        labels = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(labels, list):
            raise ValueError("--label-space-path must point to a JSON list of labels.")
        return normalize_labels(labels)

    labels: Set[str] = set()
    for split in ["train", "validation"]:
        for ex in raw[split]:
            labels.update(parse_labels(ex[cfg.labels_column]))
    return sorted(labels)


# =============================================================================
# Prompt formatting
# =============================================================================

SYSTEM_PROMPT = (
    "You are an expert patent examiner specialized in the Cooperative Patent Classification "
    "(CPC) system. Given the title and abstract of a patent, assign all relevant CPC "
    "subclasses at the four-character subclass level. Respond strictly with a JSON object "
    "of the form: {\"labels\": [\"G06F\", \"H04L\"]}. Use valid CPC subclass codes only. "
    "Do not add explanations."
)


def make_user_content(text: str) -> str:
    return (
        "Patent text (title + abstract):\n"
        "------------------------------\n"
        f"{text}\n"
        "------------------------------\n"
        "Return only the JSON object."
    )


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        rendered = []
        for m in messages:
            rendered.append(f"[{m['role'].upper()}]\n{m['content']}")
        if add_generation_prompt:
            rendered.append("[ASSISTANT]\n")
        return "\n\n".join(rendered)


def build_sft_text(example: Dict[str, Any], tokenizer, cfg: Config) -> str:
    patent_text = build_input_text(example, cfg)
    labels = parse_labels(example[cfg.labels_column])
    target_json = json.dumps({"labels": labels}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_content(patent_text)},
        {"role": "assistant", "content": target_json},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=False)


def format_datasets(raw: DatasetDict, tokenizer, cfg: Config):
    def format_split(split: str):
        def _format(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
            texts = []
            n = len(batch[cfg.title_column])
            for i in range(n):
                ex = {k: batch[k][i] for k in batch.keys()}
                texts.append(build_sft_text(ex, tokenizer, cfg))
            return {"text": texts}

        return raw[split].map(
            _format,
            batched=True,
            remove_columns=raw[split].column_names,
            desc=f"Formatting {split}",
        )

    return format_split("train"), format_split("validation")


def save_prompt_example(train_dataset, out_dir: Path) -> None:
    if len(train_dataset) > 0:
        (out_dir / "formatted_train_example.txt").write_text(train_dataset[0]["text"], encoding="utf-8")


# =============================================================================
# Model, LoRA, trainer compatibility
# =============================================================================

def load_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=False, local_files_only=cfg.offline)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(cfg: Config, dtype: torch.dtype):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. This script expects a GPU for non-quantized 7B LoRA fine-tuning.")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=cfg.offline,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def make_lora_config(cfg: Config) -> LoraConfig:
    targets = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )


def make_training_args(cfg: Config, dtype: torch.dtype) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    supports_eval_strategy = "eval_strategy" in sig.parameters

    args = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=cfg.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=True,
    )

    if supports_eval_strategy:
        args["eval_strategy"] = "steps"
    else:
        args["evaluation_strategy"] = "steps"

    return TrainingArguments(**args)


def make_sft_trainer(model, tokenizer, train_dataset, eval_dataset, peft_config, training_args, cfg: Config):
    sig = inspect.signature(SFTTrainer.__init__)
    kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        ],
    )

    # TRL has changed argument names across versions. Handle common variants.
    if "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in sig.parameters:
        kwargs["dataset_text_field"] = "text"
    if "packing" in sig.parameters:
        kwargs["packing"] = False
    if "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = cfg.max_seq_length

    return SFTTrainer(**kwargs)


# =============================================================================
# Training logs and CodeCarbon
# =============================================================================

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step}
        for key, value in logs.items():
            if isinstance(value, (int, float, np.floating)):
                record[key] = float(value)
        self.records.append(record)


def save_learning_curve(log_history: List[Dict[str, Any]], out_dir: Path) -> None:
    if not log_history:
        return
    df = pd.DataFrame(log_history)
    df.to_csv(out_dir / "learning_curve_logs.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        if "loss" in df.columns:
            train_df = df.dropna(subset=["loss"])
            plt.plot(train_df["step"], train_df["loss"], label="train_loss")
        if "eval_loss" in df.columns:
            eval_df = df.dropna(subset=["eval_loss"])
            plt.plot(eval_df["step"], eval_df["eval_loss"], label="eval_loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "learning_curve_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"[warn] Could not save learning curve plot: {e}")


def train_model(model, tokenizer, train_dataset, eval_dataset, cfg: Config, dtype: torch.dtype):
    out_dir = Path(cfg.output_dir)
    training_args = make_training_args(cfg, dtype)
    peft_config = make_lora_config(cfg)
    loss_callback = LossHistoryCallback()
    trainer = make_sft_trainer(model, tokenizer, train_dataset, eval_dataset, peft_config, training_args, cfg)
    trainer.add_callback(loss_callback)

    tracker = None
    emissions_kg = None
    if cfg.use_codecarbon:
        if EmissionsTracker is None:
            print("[warn] codecarbon is not installed; emissions tracking disabled.")
        else:
            tracker = EmissionsTracker(
                project_name=cfg.codecarbon_project_name,
                output_dir=cfg.output_dir,
                measure_power_secs=cfg.codecarbon_measure_power_secs,
                save_to_file=True,
                log_level="error",
            )

    start = time.time()
    if tracker is not None:
        tracker.start()
    train_result = trainer.train()
    if tracker is not None:
        emissions_kg = tracker.stop()
    end = time.time()

    total_time_sec = end - start
    summary = {
        "total_time_sec": total_time_sec,
        "time_per_training_example_sec": total_time_sec / max(len(train_dataset), 1),
        "total_emissions_kg": emissions_kg,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "num_train_examples": len(train_dataset),
        "train_result": train_result.metrics if hasattr(train_result, "metrics") else {},
    }
    save_json(out_dir / "training_summary.json", summary)
    save_learning_curve(trainer.state.log_history, out_dir)
    return trainer


def save_adapter_and_tokenizer(trainer, tokenizer, cfg: Config) -> Path:
    adapter_path = Path(cfg.output_dir) / cfg.adapter_subdir
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    return adapter_path


# =============================================================================
# Generation evaluation
# =============================================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"raw_output": raw}


def prediction_to_labels(pred: Dict[str, Any]) -> List[str]:
    if isinstance(pred, dict) and isinstance(pred.get("labels"), list):
        return normalize_labels(pred["labels"])
    return []


def labels_to_binary(labels: List[str], label_to_id: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(label_to_id), dtype=np.int32)
    for lab in labels:
        if lab in label_to_id:
            y[label_to_id[lab]] = 1
    return y


def load_adapter_for_inference(adapter_path: Path, cfg: Config, dtype: torch.dtype):
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=cfg.offline,
    )
    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=False, local_files_only=cfg.offline)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = PeftModel.from_pretrained(base_model, str(adapter_path), local_files_only=True).eval()
    return model, tok


def build_generation_prompt(example: Dict[str, Any], tokenizer, cfg: Config) -> str:
    text = build_input_text(example, cfg)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_content(text)},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=True)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], cfg: Config) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_seq_length,
    )
    prompt_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model_device(model)) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][prompt_len:]
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return decoded


def evaluate_on_dev_by_generation(model, tokenizer, raw: DatasetDict, label_space: List[str], cfg: Config) -> Dict[str, Any]:
    dev = raw["validation"]
    if cfg.max_eval_examples is not None:
        dev = dev.select(range(min(cfg.max_eval_examples, len(dev))))

    label_to_id = {lab: i for i, lab in enumerate(label_space)}
    y_true, y_pred, rows = [], [], []

    for start in tqdm(range(0, len(dev), cfg.generation_batch_size), desc="Dev generation evaluation"):
        batch = dev[start:start + cfg.generation_batch_size]
        batch_examples = []
        prompts = []
        n = len(batch[cfg.title_column])
        for j in range(n):
            ex = {k: batch[k][j] for k in batch.keys()}
            batch_examples.append(ex)
            prompts.append(build_generation_prompt(ex, tokenizer, cfg))

        raw_outputs = generate_batch(model, tokenizer, prompts, cfg)

        for offset, (ex, raw_output) in enumerate(zip(batch_examples, raw_outputs)):
            idx = start + offset
            gold_labels = parse_labels(ex[cfg.labels_column])
            pred_obj = extract_json_object(raw_output)
            pred_labels = prediction_to_labels(pred_obj)

            y_true.append(labels_to_binary(gold_labels, label_to_id))
            y_pred.append(labels_to_binary(pred_labels, label_to_id))
            rows.append({
                "idx": idx,
                "title": ex.get(cfg.title_column, ""),
                "gold_labels": json.dumps(gold_labels),
                "pred_labels": json.dumps(pred_labels),
                "raw_generation": raw_output,
                "parsed_prediction": json.dumps(pred_obj, ensure_ascii=False),
                "num_gold": len(gold_labels),
                "num_pred": len(pred_labels),
                "exact_match": int(set(gold_labels) == set(pred_labels)),
            })

    y_true_arr = np.vstack(y_true) if y_true else np.zeros((0, len(label_space)))
    y_pred_arr = np.vstack(y_pred) if y_pred else np.zeros((0, len(label_space)))

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    samples_f1 = f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0)

    metrics = {
        "num_dev_examples": len(rows),
        "num_labels": len(label_space),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "samples_f1": float(samples_f1),
        "exact_match": float(np.mean([r["exact_match"] for r in rows])) if rows else 0.0,
        "avg_gold_labels": float(np.mean([r["num_gold"] for r in rows])) if rows else 0.0,
        "avg_pred_labels": float(np.mean([r["num_pred"] for r in rows])) if rows else 0.0,
        "empty_pred_rate": float(np.mean([r["num_pred"] == 0 for r in rows])) if rows else 0.0,
    }

    out_dir = Path(cfg.output_dir)
    pd.DataFrame(rows).to_csv(out_dir / "dev_generation_predictions.csv", index=False)
    save_json(out_dir / "dev_generation_metrics.json", metrics)
    save_json(out_dir / "label_space.json", label_space)
    return metrics


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    configure_environment(cfg)
    set_all_seeds(cfg.seed)

    output_dir = Path(cfg.output_dir)
    prepare_output_dir(output_dir, cfg.overwrite)
    save_json(output_dir / "run_environment.json", environment_report(cfg))

    dtype = dtype_for_training()
    print(f"Using dtype: {dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("[data] loading datasets")
    raw = load_raw_datasets(cfg)
    label_space = collect_label_space(raw, cfg)
    save_json(output_dir / "data_summary.json", {
        "train_examples": len(raw["train"]),
        "dev_examples": len(raw["validation"]),
        "num_labels": len(label_space),
        "avg_train_labels": float(np.mean([len(parse_labels(ex[cfg.labels_column])) for ex in raw["train"]])),
        "avg_dev_labels": float(np.mean([len(parse_labels(ex[cfg.labels_column])) for ex in raw["validation"]])),
    })

    print("[model] loading tokenizer and base model")
    tokenizer = load_tokenizer(cfg)
    train_dataset, eval_dataset = format_datasets(raw, tokenizer, cfg)
    save_prompt_example(train_dataset, output_dir)

    model = load_base_model(cfg, dtype)

    print("[train] starting LoRA fine-tuning")
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, cfg, dtype)

    print("[save] saving adapter")
    adapter_path = save_adapter_and_tokenizer(trainer, tokenizer, cfg)
    save_json(output_dir / "adapter_summary.json", {"adapter_path": str(adapter_path)})

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[eval] loading adapter for generation evaluation")
    ft_model, inf_tokenizer = load_adapter_for_inference(adapter_path, cfg, dtype)
    metrics = evaluate_on_dev_by_generation(ft_model, inf_tokenizer, raw, label_space, cfg)

    print("\n=== Dev generation metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
