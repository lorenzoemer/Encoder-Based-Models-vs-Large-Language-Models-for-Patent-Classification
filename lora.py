"""
LoRA hyperparameter sweep for generative CPC patent classification.

The script fine-tunes an instruction-tuned language model (Qwen 3.5-9B) to generate one to
seven CPC subclass codes from patent titles and abstracts. It compares several
LoRA ranks, scaling factors, learning rates, and input lengths.

It expects training and validation TSV files, a CPC label-mapping JSON file,
a base-model path, and an output directory. For each configuration, it saves
the best adapter, validation predictions, and Micro-F1, Macro-F1, hierarchical
F1, validation loss, and training metadata. The final configurations are ranked
by validation Micro-F1 and Macro-F1.
"""


#!/usr/bin/env python3
import argparse
import ast
import csv
import gc
import inspect
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
NUM_EPOCHS = 5.0
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
GENERATION_BATCH_SIZE = 2
EVAL_STEPS = 250
SAVE_STEPS = 250
LOGGING_STEPS = 25
EARLY_STOPPING_PATIENCE = 2
MAX_NEW_TOKENS = 64
GENERATION_EVAL_SIZE = 2000

SWEEP = [
    {"name": "r16_a32_lr1e-4_len512", "r": 16, "alpha": 32, "lr": 1e-4, "max_length": 512},
    {"name": "r16_a32_lr1e-4_len1024", "r": 16, "alpha": 32, "lr": 1e-4, "max_length": 1024},
    {"name": "r16_a32_lr2e-4_len2048", "r": 16, "alpha": 32, "lr": 2e-4, "max_length": 2048},
    {"name": "r32_a64_lr2e-4_len2048", "r": 32, "alpha": 64, "lr": 2e-4, "max_length": 2048},
]

SYSTEM_PROMPT = (
    "You are an expert patent examiner specialized in the Cooperative Patent Classification (CPC) system. "
    "Given the title and abstract of a patent, assign all relevant CPC subclasses. "
    'Respond strictly with a JSON object of the form: {"labels": ["CPC1", "CPC2", ...]}. '
    "Use valid four-character CPC subclass codes only. Return between 1 and 7 labels. "
    "Do not provide explanations or reasoning."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def normalize_code(value):
    code = re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper().split("/")[0])
    if len(code) >= 4 and code[0].isalpha() and code[1:3].isdigit() and code[3].isalpha():
        return code[:4]
    return ""


def parse_labels(raw):
    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif isinstance(raw, str):
        values = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, dict):
                    parsed = parsed.get("labels")
                if isinstance(parsed, (list, tuple, set)):
                    values = list(parsed)
                    break
            except Exception:
                pass
        if values is None:
            values = [item for item in re.split(r"[,;|]", raw) if item.strip()]
    else:
        values = [raw]

    output = []
    seen = set()
    for value in values:
        code = normalize_code(value)
        if code and code not in seen:
            seen.add(code)
            output.append(code)
    return output


def read_tsv(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}. {abstract}".strip(". ")
            labels = parse_labels(row.get("labels"))
            if text and labels:
                rows.append(
                    {
                        "id": str(row.get("patent_id") or row.get("id") or index),
                        "text": text,
                        "labels": labels,
                    }
                )
    return rows


def load_label_mapping(path):
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(mapping, list):
        raw_labels = mapping
    elif isinstance(mapping.get("labels"), list):
        raw_labels = mapping["labels"]
    elif isinstance(mapping.get("label2id"), dict):
        raw_labels = [key for key, value in sorted(mapping["label2id"].items(), key=lambda item: int(item[1]))]
    elif isinstance(mapping.get("id2label"), dict):
        raw_labels = [mapping["id2label"][str(index)] for index in sorted(map(int, mapping["id2label"]))]
    elif isinstance(mapping.get("label_to_id"), dict):
        raw_labels = [key for key, value in sorted(mapping["label_to_id"].items(), key=lambda item: int(item[1]))]
    elif isinstance(mapping.get("id_to_label"), dict):
        raw_labels = [mapping["id_to_label"][str(index)] for index in sorted(map(int, mapping["id_to_label"]))]
    else:
        raise ValueError("Unsupported label mapping format")

    labels = []
    seen = set()
    for value in raw_labels:
        code = normalize_code(value)
        if code and code not in seen:
            seen.add(code)
            labels.append(code)
    return labels


def build_messages(text):
    return [
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


def serialize_answer(labels):
    return json.dumps({"labels": list(labels)[:7]}, separators=(",", ":"))


def apply_chat_template(tokenizer, messages, add_generation_prompt):
    variants = (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    )
    for extra in variants:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra,
            )
        except TypeError:
            pass
    raise RuntimeError("Unable to apply the model chat template")


class PatentDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt_messages = build_messages(row["text"])
        full_messages = prompt_messages + [
            {"role": "assistant", "content": serialize_answer(row["labels"])}
        ]

        prompt_text = apply_chat_template(self.tokenizer, prompt_messages, True)
        full_text = apply_chat_template(self.tokenizer, full_messages, False)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        if all(value == -100 for value in labels):
            answer_ids = self.tokenizer(
                serialize_answer(row["labels"]),
                add_special_tokens=False,
            )["input_ids"]
            keep = min(len(answer_ids), len(labels))
            labels[-keep:] = encoded["input_ids"][-keep:]

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        max_length = ((max_length + 7) // 8) * 8
        pad_id = self.tokenizer.pad_token_id

        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            values = []
            for feature in features:
                if key == "input_ids":
                    pad_value = pad_id
                elif key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = -100
                values.append(
                    feature[key] + [pad_value] * (max_length - len(feature[key]))
                )
            batch[key] = torch.tensor(values, dtype=torch.long)
        return batch


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    errors = []
    for model_class in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = model_class.from_pretrained(model_name, **kwargs)
            return model.to("cuda")
        except Exception as error:
            errors.append(f"{model_class.__name__}: {error}")
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError("\n".join(errors))


def add_lora(model, configuration):
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    lora_configuration = LoraConfig(
        r=configuration["r"],
        lora_alpha=configuration["alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_configuration)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def create_training_arguments(output_dir, learning_rate):
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "optim": "adamw_torch",
        "seed": SEED,
        "data_seed": SEED,
    }

    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def extract_labels(text, valid_labels):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    values = []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            values = parsed.get("labels", [])
    except Exception:
        pass

    if not values:
        for substring in re.findall(r"\{.*?\}", text, flags=re.S):
            try:
                parsed = json.loads(substring)
                if isinstance(parsed, dict) and isinstance(parsed.get("labels"), list):
                    values = parsed["labels"]
                    break
            except Exception:
                pass

    if not values:
        values = re.findall(r"\b[A-HY][0-9]{2}[A-Z]\b", text.upper())

    output = []
    seen = set()
    for value in values:
        code = normalize_code(value)
        if code in valid_labels and code not in seen:
            seen.add(code)
            output.append(code)

    return output[:7]


def compute_metrics(gold_labels, predicted_labels, label_space):
    binarizer = MultiLabelBinarizer(classes=label_space)
    gold = binarizer.fit_transform(gold_labels)
    predicted = binarizer.transform(predicted_labels)

    micro = precision_recall_fscore_support(
        gold,
        predicted,
        average="micro",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        gold,
        predicted,
        average="macro",
        zero_division=0,
    )

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_set, predicted_set in zip(gold_labels, predicted_labels):
        gold_hierarchy = {
            ancestor
            for code in gold_set
            for ancestor in (code[0], code[:3], code)
        }
        predicted_hierarchy = {
            ancestor
            for code in predicted_set
            for ancestor in (code[0], code[:3], code)
        }

        true_positive += len(gold_hierarchy & predicted_hierarchy)
        false_positive += len(predicted_hierarchy - gold_hierarchy)
        false_negative += len(gold_hierarchy - predicted_hierarchy)

    hierarchical_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    hierarchical_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    hierarchical_f1 = (
        2
        * hierarchical_precision
        * hierarchical_recall
        / (hierarchical_precision + hierarchical_recall)
        if hierarchical_precision + hierarchical_recall
        else 0.0
    )

    return {
        "micro_f1": float(micro[2]),
        "macro_f1": float(macro[2]),
        "hierarchical_f1": float(hierarchical_f1),
    }


@torch.inference_mode()
def generation_evaluation(
    model,
    tokenizer,
    rows,
    label_space,
    max_length,
    output_path,
):
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    predictions = []
    records = []
    valid_labels = set(label_space)

    for start in tqdm(
        range(0, len(rows), GENERATION_BATCH_SIZE),
        desc="Validation generation",
    ):
        batch = rows[start : start + GENERATION_BATCH_SIZE]
        prompts = [
            apply_chat_template(tokenizer, build_messages(row["text"]), True)
            for row in batch
        ]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}

        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_width = encoded["input_ids"].shape[1]

        for row_index, row in enumerate(batch):
            raw_output = tokenizer.decode(
                generated[row_index, prompt_width:],
                skip_special_tokens=True,
            ).strip()
            predicted = extract_labels(raw_output, valid_labels)
            predictions.append(predicted)
            records.append(
                {
                    "id": row["id"],
                    "gold": row["labels"],
                    "predicted": predicted,
                    "raw_output": raw_output,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    tokenizer.padding_side = "right"
    model.config.use_cache = False

    return compute_metrics(
        [row["labels"] for row in rows],
        predictions,
        label_space,
    )


def run_configuration(
    model_name,
    configuration,
    training_rows,
    validation_rows,
    generation_rows,
    label_space,
    output_root,
):
    run_directory = output_root / configuration["name"]
    checkpoint_directory = run_directory / "checkpoints"
    adapter_directory = run_directory / "best_adapter"
    run_directory.mkdir(parents=True, exist_ok=True)

    seed_everything()

    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name)
    model = add_lora(base_model, configuration)

    trainer = Trainer(
        model=model,
        args=create_training_arguments(
            checkpoint_directory,
            configuration["lr"],
        ),
        train_dataset=PatentDataset(
            training_rows,
            tokenizer,
            configuration["max_length"],
        ),
        eval_dataset=PatentDataset(
            validation_rows,
            tokenizer,
            configuration["max_length"],
        ),
        data_collator=DataCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.001,
            )
        ],
    )

    start_time = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start_time

    trainer.save_model(str(adapter_directory))
    tokenizer.save_pretrained(str(adapter_directory))

    evaluation = trainer.evaluate()
    generation_metrics = generation_evaluation(
        trainer.model,
        tokenizer,
        generation_rows,
        label_space,
        configuration["max_length"],
        run_directory / "validation_predictions.jsonl",
    )

    result = {
        "run_name": configuration["name"],
        "lora_rank": configuration["r"],
        "lora_alpha": configuration["alpha"],
        "learning_rate": configuration["lr"],
        "maximum_length": configuration["max_length"],
        "epoch_reached": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "training_seconds": training_seconds,
        "validation_loss": evaluation.get("eval_loss"),
        "validation_micro_f1": generation_metrics["micro_f1"],
        "validation_macro_f1": generation_metrics["macro_f1"],
        "validation_hierarchical_f1": generation_metrics["hierarchical_f1"],
    }

    (run_directory / "results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output.mkdir(parents=True, exist_ok=True)

    training_rows = read_tsv(args.train)
    validation_rows = read_tsv(args.validation)
    label_space = load_label_mapping(args.label_mapping)

    generator = np.random.default_rng(SEED)
    selected_indices = generator.choice(
        len(validation_rows),
        size=min(GENERATION_EVAL_SIZE, len(validation_rows)),
        replace=False,
    )
    generation_rows = [
        validation_rows[int(index)]
        for index in selected_indices
    ]

    results = []

    for configuration in SWEEP:
        result = run_configuration(
            args.model,
            configuration,
            training_rows,
            validation_rows,
            generation_rows,
            label_space,
            args.output,
        )
        results.append(result)

        pd.DataFrame(results).to_csv(
            args.output / "sweep_results.csv",
            index=False,
        )
        (args.output / "sweep_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    ranked = pd.DataFrame(results).sort_values(
        ["validation_micro_f1", "validation_macro_f1"],
        ascending=False,
    )
    ranked.to_csv(
        args.output / "sweep_results_ranked.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
