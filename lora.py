#!/usr/bin/env python3
"""
LoRA fine-tuning for CPC multi-label patent classification with an instruction-tuned causal LM.

Inputs:
- Train/validation TSV files with columns: title, abstract, labels
- A local base model directory

Outputs:
- <output_dir>/adapter/ (LoRA adapter + tokenizer files)
- Training logs/checkpoints under <output_dir> (Transformers)
- CodeCarbon emissions file(s) under <output_dir> (optional)
"""

import os
import json
import ast
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


SYSTEM_PROMPT = (
    "You are an expert patent examiner specialized in the Cooperative Patent Classification (CPC) system. "
    "Given the title and abstract of a patent, you must assign all relevant CPC subclasses. "
    "Respond strictly with a JSON object of the form: {\"labels\": [\"CPC1\", \"CPC2\", ...]}. "
    "Use valid CPC subclass codes only. Do not add explanations."
)


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
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [p.strip() for p in s.split(";") if p.strip()]

    return [str(raw).strip()]


def build_text(title: str, abstract: str) -> str:
    return ((title or "").strip() + ". " + (abstract or "").strip()).strip()


def build_sft_example(example: Dict[str, Any], tokenizer: AutoTokenizer, title_col: str, abstract_col: str, labels_col: str) -> str:
    text = build_text(example.get(title_col, "") or "", example.get(abstract_col, "") or "")
    labels = parse_labels(example.get(labels_col))
    target_json = json.dumps({"labels": labels}, ensure_ascii=False)

    user_content = (
        "Patent text (title + abstract):\n"
        "------------------------------\n"
        f"{text}\n"
        "------------------------------\n"
        "Return only the JSON object."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_json},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_train_val_datasets(train_path: Path, val_path: Path) -> DatasetDict:
    data_files = {"train": str(train_path), "validation": str(val_path)}
    return load_dataset("csv", data_files=data_files, delimiter="\t")


def format_datasets(
    raw: DatasetDict,
    tokenizer: AutoTokenizer,
    title_col: str,
    abstract_col: str,
    labels_col: str,
):
    def _format_split(split: str):
        def _format(batch):
            n = len(batch[title_col])
            texts = []
            for i in range(n):
                ex = {k: batch[k][i] for k in batch.keys()}
                texts.append(build_sft_example(ex, tokenizer, title_col, abstract_col, labels_col))
            return {"text": texts}

        return raw[split].map(_format, batched=True, remove_columns=raw[split].column_names)

    return _format_split("train"), _format_split("validation")


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def load_base_model(model_path: Path, dtype: torch.dtype, local_only: bool):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {model_path}")

    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_only,
    )


def load_tokenizer(model_path: Path, local_only: bool) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=False, local_files_only=local_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    dtype: torch.dtype,
    use_codecarbon: bool,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=make_lora_config(),
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
        args=args,
    )

    tracker = None
    if use_codecarbon and EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name="lora_training",
            output_dir=str(output_dir),
            measure_power_secs=10,
            save_to_file=True,
            log_level="error",
        )

    start = time.time()
    if tracker is not None:
        tracker.start()

    result = trainer.train()

    emissions_kg = tracker.stop() if tracker is not None else None
    total_time_sec = time.time() - start

    summary = {
        "num_train_examples": int(len(train_dataset)),
        "total_time_sec": float(total_time_sec),
        "time_per_example_sec": float(total_time_sec / len(train_dataset)) if len(train_dataset) else None,
        "total_co2_emissions_kg": float(emissions_kg) if emissions_kg is not None else None,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return trainer, result


def save_adapter_and_tokenizer(trainer, tokenizer, output_dir: Path, adapter_subdir: str) -> Path:
    adapter_path = output_dir / adapter_subdir
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    return adapter_path


def load_finetuned_for_inference(base_model_path: Path, adapter_path: Path, dtype: torch.dtype, local_only: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_only,
    )
    tok = load_tokenizer(base_model_path, local_only=local_only)
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).eval()
    return model, tok


def classify_one(model, tokenizer, title: str, abstract: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    text = build_text(title, abstract)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Patent text (title + abstract):\n"
                "------------------------------\n"
                f"{text}\n"
                "------------------------------\n"
                "Return only the JSON object."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    try:
        return json.loads(generated)
    except Exception:
        return {"raw_output": generated}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=Path, required=True)

    ap.add_argument("--train_path", type=Path, required=True)
    ap.add_argument("--val_path", type=Path, required=True)

    ap.add_argument("--title_col", type=str, default="title")
    ap.add_argument("--abstract_col", type=str, default="abstract")
    ap.add_argument("--labels_col", type=str, default="labels")

    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--adapter_subdir", type=str, default="adapter")

    ap.add_argument("--local_only", action="store_true", default=True)
    ap.add_argument("--use_codecarbon", action="store_true", default=True)

    ap.add_argument("--smoke_test", action="store_true", default=False)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not args.train_path.is_file():
        raise FileNotFoundError(f"Missing train file: {args.train_path}")
    if not args.val_path.is_file():
        raise FileNotFoundError(f"Missing validation file: {args.val_path}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = load_tokenizer(args.model_path, local_only=args.local_only)
    raw = load_train_val_datasets(args.train_path, args.val_path)

    train_dataset, eval_dataset = format_datasets(
        raw=raw,
        tokenizer=tokenizer,
        title_col=args.title_col,
        abstract_col=args.abstract_col,
        labels_col=args.labels_col,
    )

    model = load_base_model(args.model_path, dtype=dtype, local_only=args.local_only)

    trainer, _ = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        dtype=dtype,
        use_codecarbon=args.use_codecarbon,
    )

    adapter_path = save_adapter_and_tokenizer(trainer, tokenizer, args.output_dir, args.adapter_subdir)

    if args.smoke_test:
        ft_model, ft_tok = load_finetuned_for_inference(
            base_model_path=args.model_path,
            adapter_path=adapter_path,
            dtype=dtype,
            local_only=args.local_only,
        )
        out = classify_one(ft_model, ft_tok, title="Example title", abstract="Example abstract")
        (args.output_dir / "smoke_test_output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
