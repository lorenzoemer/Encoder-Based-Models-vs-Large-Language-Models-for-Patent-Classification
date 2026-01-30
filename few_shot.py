#!/usr/bin/env python3
"""
Few-shot evaluation of LLMs for CPC multi-label patent classification.

FEW-SHOT setting:
- Few-shot = 5 labeled examples total:
  - 2 static examples (fixed training indices)
  - 3 dynamic examples retrieved from TRAIN by semantic similarity (E5-base-v2 embeddings, cosine / dot-product on normalized vectors)
- No RAG (no CPC-definition retrieval)
- Deterministic decoding (do_sample=False)
- Output constraint: 1 to 7 CPC subclasses (4-character level)
- Parsing: strict JSON {"labels": [...]} with fallback to extracting first JSON object
- If parsing yields empty: minimal non-empty fallback (first CPC-like code found in generation)

Inputs:
- TRAIN TSV: columns title, abstract, labels
- TEST TSV: columns title, abstract, labels
- Local embedding model directory (E5-base-v2)

Outputs per model under: <output_dir>/<model_name>/
- metrics.json
- predictions.jsonl (includes prompt and retrieval metadata: static_ids, dynamic_ids)
- per_label_section.csv
- per_label_class.csv
- per_label_subclass.csv
- codecarbon emissions file(s) (optional)
"""

import os
import re
import json
import ast
import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


SYSTEM_PROMPT = (
    "You are an expert patent examiner for the Cooperative Patent Classification (CPC) system. "
    "Your goal is HIGH RECALL classification: it is worse to miss a relevant CPC subclass "
    "than to include a marginally relevant one. "
    "You assign MULTIPLE CPC subclasses (multi-label) to each patent. "
    "Return ONLY a strict JSON object with a single key \"labels\", "
    "whose value is a list of CPC subclass codes (e.g. [\"G06Q\", \"Y02D\"]). "
    "Return JSON ONLY (no extra text)."
)

USER_PROMPT_TEMPLATE = (
    "You will see some labeled examples first.\n\n"
    "FEW-SHOT EXAMPLES:\n"
    "{fewshot_block}\n\n"
    "NOW CLASSIFY THIS PATENT:\n"
    "----------------------------------------\n"
    "{patent_text}\n"
    "----------------------------------------\n\n"
    "TASK:\n"
    "- Assign ALL relevant CPC subclasses (4-character level like A01B, G06F, Y02D).\n"
    "- Output between 1 and {max_labels} CPC subclass codes.\n"
    "- Do NOT invent codes that are not real CPC subclasses.\n\n"
    "OUTPUT FORMAT (STRICT JSON ONLY):\n"
    "{{\"labels\": [\"G06F\", \"H04L\"]}}\n"
    "JSON ONLY.\n"
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
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
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


def parse_predicted_labels(text: str, max_labels: int) -> List[str]:
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


def enforce_nonempty(pred: List[str], raw_text: str) -> List[str]:
    if pred:
        return pred
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


def format_fewshot_example(ex_text: str, ex_labels: List[str], idx: int, max_labels: int) -> str:
    labs = [normalize_cpc_label(x) for x in ex_labels if normalize_cpc_label(x)]
    labs = _dedupe_keep_order([x for x in labs if x])[:max_labels]
    return (
        f"Example {idx}:\n"
        f"PATENT TEXT:\n{ex_text}\n"
        f"OUTPUT:\n{{\"labels\": {json.dumps(labs)}}}\n"
    )


def load_encoder(embed_model_path: Path, local_only: bool):
    if not embed_model_path.is_dir():
        raise FileNotFoundError(f"Missing embedding model directory: {embed_model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(embed_model_path), local_files_only=local_only)
    mdl = AutoModel.from_pretrained(str(embed_model_path), local_files_only=local_only).to(device)
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
    model: AutoModel,
    tokenizer: AutoTokenizer,
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

    out = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i : i + batch_size]
        enc = tokenizer(
            bt,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        res = model(**enc)
        emb = _mean_pool(res.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def retrieve_dynamic_examples(
    query_texts: List[str],
    train_embeds_cpu: torch.Tensor,
    train_texts: List[str],
    train_labels: List[List[str]],
    embed_model: AutoModel,
    embed_tokenizer: AutoTokenizer,
    embed_device: str,
    k: int,
    exclude_ids: Optional[List[int]],
    query_emb_batch: int,
    max_labels_per_example: int,
) -> Tuple[List[List[int]], List[List[str]]]:
    q_emb_cpu = encode_texts_e5(
        query_texts,
        embed_model,
        embed_tokenizer,
        embed_device,
        batch_size=min(query_emb_batch, max(1, len(query_texts))),
        prefix="query",
    )
    q = q_emb_cpu.to(embed_device)
    T = train_embeds_cpu.to(embed_device)

    sims = q @ T.T
    sims = sims.cpu()

    dyn_ids_all: List[List[int]] = []
    dyn_blocks: List[List[str]] = []

    excl = set(exclude_ids or [])

    for i in range(sims.size(0)):
        row = sims[i].clone()
        if excl:
            for eid in excl:
                if 0 <= eid < row.numel():
                    row[eid] = -1e9

        top_idx = torch.topk(row, k=min(k, row.numel()), dim=0).indices.tolist()
        top_idx = [int(x) for x in top_idx]

        dyn_ids_all.append(top_idx)
        blocks = []
        for j, tid in enumerate(top_idx, start=1):
            blocks.append(
                format_fewshot_example(
                    ex_text=train_texts[tid],
                    ex_labels=train_labels[tid],
                    idx=j,
                    max_labels=max_labels_per_example,
                )
            )
        dyn_blocks.append(blocks)

    return dyn_ids_all, dyn_blocks


def load_llm(model_path: Path, local_only: bool):
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing model directory: {model_path}")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=local_only, use_fast=False)
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
    fewshot_blocks: List[str],
    max_new_tokens: int,
    max_labels: int,
) -> Tuple[List[List[str]], List[str], List[str]]:
    prompts: List[str] = []
    for title, abstract, fs_block in zip(titles, abstracts, fewshot_blocks):
        patent_text = build_text(title, abstract)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            fewshot_block=fs_block,
            patent_text=patent_text,
            max_labels=max_labels,
        )
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
        labs = enforce_nonempty(labs, gen_text)
        preds.append(labs[:max_labels])

    return preds, raw_texts, prompts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_path", type=Path, required=True)
    ap.add_argument("--test_path", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--embed_model_path", type=Path, required=True)

    ap.add_argument("--title_col", type=str, default="title")
    ap.add_argument("--abstract_col", type=str, default="abstract")
    ap.add_argument("--labels_col", type=str, default="labels")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_eval", type=int, default=None)

    ap.add_argument("--static_k", type=int, default=2)
    ap.add_argument("--dynamic_k", type=int, default=3)
    ap.add_argument("--static_example_ids", type=int, nargs="+", required=True)

    ap.add_argument("--cpc_emb_batch", type=int, default=64)
    ap.add_argument("--query_emb_batch", type=int, default=16)

    ap.add_argument("--max_labels", type=int, default=7)

    ap.add_argument("--local_only", action="store_true", default=True)
    ap.add_argument("--use_codecarbon", action="store_true", default=True)

    ap.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "MODEL_PATH"),
        required=True,
        help="Repeatable. Example: --model qwen models/Qwen2.5-7B-Instruct",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not args.train_path.is_file():
        raise FileNotFoundError(f"Missing train file: {args.train_path}")
    if not args.test_path.is_file():
        raise FileNotFoundError(f"Missing test file: {args.test_path}")
    if not args.embed_model_path.is_dir():
        raise FileNotFoundError(f"Missing embedding model directory: {args.embed_model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("csv", data_files={"train": str(args.train_path)}, delimiter="\t")["train"]
    test_ds = load_dataset("csv", data_files={"test": str(args.test_path)}, delimiter="\t")["test"]
    if args.max_eval is not None:
        test_ds = test_ds.select(range(min(int(args.max_eval), len(test_ds))))

    static_ids = [int(x) for x in args.static_example_ids[: args.static_k]]
    for sid in static_ids:
        if sid < 0 or sid >= len(train_ds):
            raise ValueError(f"Invalid static_example_id {sid} (train size={len(train_ds)})")

    train_titles = [ex.get(args.title_col, "") or "" for ex in train_ds]
    train_abstracts = [ex.get(args.abstract_col, "") or "" for ex in train_ds]
    train_texts = [build_text(t, a) for t, a in zip(train_titles, train_abstracts)]

    train_labels: List[List[str]] = []
    for ex in train_ds:
        labs = [normalize_cpc_label(x) for x in parse_labels(ex.get(args.labels_col))]
        labs = [x for x in labs if x]
        train_labels.append(labs)

    embed_model, embed_tok, embed_dev = load_encoder(args.embed_model_path, local_only=args.local_only)

    train_embeds_cpu = encode_texts_e5(
        texts=train_texts,
        model=embed_model,
        tokenizer=embed_tok,
        device=embed_dev,
        batch_size=args.cpc_emb_batch,
        prefix="passage",
    ).cpu()

    test_titles: List[str] = []
    test_abstracts: List[str] = []
    test_texts: List[str] = []
    gold_labels: List[List[str]] = []

    for ex in test_ds:
        t = ex.get(args.title_col, "") or ""
        a = ex.get(args.abstract_col, "") or ""
        test_titles.append(t)
        test_abstracts.append(a)
        test_texts.append(build_text(t, a))

        labs = [normalize_cpc_label(x) for x in parse_labels(ex.get(args.labels_col))]
        labs = [x for x in labs if x]
        gold_labels.append(labs)

    test_label_universe = sorted(set(l for labs in gold_labels for l in labs))
    if not test_label_universe:
        raise ValueError("No gold labels found after parsing test labels.")

    static_blocks: List[str] = []
    for k, sid in enumerate(static_ids, start=1):
        static_blocks.append(
            format_fewshot_example(train_texts[sid], train_labels[sid], idx=k, max_labels_per_example=args.max_labels)
        )
    static_block_text = "\n".join(static_blocks).strip()

    dyn_ids_all, dyn_blocks = retrieve_dynamic_examples(
        query_texts=test_texts,
        train_embeds_cpu=train_embeds_cpu,
        train_texts=train_texts,
        train_labels=train_labels,
        embed_model=embed_model,
        embed_tokenizer=embed_tok,
        embed_device=embed_dev,
        k=args.dynamic_k,
        exclude_ids=static_ids,
        query_emb_batch=args.query_emb_batch,
        max_labels_per_example=args.max_labels,
    )

    fewshot_blocks: List[str] = []
    retrieval_meta_all: List[Dict[str, Any]] = []

    for i in range(len(test_texts)):
        dyn = "\n".join(dyn_blocks[i]).strip()
        fs = (static_block_text + "\n\n" + dyn).strip() if dyn else static_block_text
        fewshot_blocks.append(fs)
        retrieval_meta_all.append({"static_ids": static_ids, "dynamic_ids": dyn_ids_all[i]})

    for model_name, model_path_str in args.model:
        model_path = Path(model_path_str)
        outdir = args.output_dir / model_name
        outdir.mkdir(parents=True, exist_ok=True)

        model, tok = load_llm(model_path, local_only=args.local_only)

        tracker = None
        if args.use_codecarbon and EmissionsTracker is not None:
            tracker = EmissionsTracker(
                project_name=f"few_shot_{model_name}",
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
        prompts_all: List[str] = []

        for i in tqdm(range(0, len(test_titles), args.batch_size), desc=f"Inference [{model_name}]"):
            bt = test_titles[i : i + args.batch_size]
            ba = test_abstracts[i : i + args.batch_size]
            bfs = fewshot_blocks[i : i + args.batch_size]

            batch_preds, batch_raw, batch_prompts = generate_batch(
                model=model,
                tokenizer=tok,
                titles=bt,
                abstracts=ba,
                fewshot_blocks=bfs,
                max_new_tokens=args.max_new_tokens,
                max_labels=args.max_labels,
            )
            pred_labels.extend(batch_preds)
            raw_generations.extend(batch_raw)
            prompts_all.extend(batch_prompts)

        emissions_kg = tracker.stop() if tracker is not None else None
        total_time_sec = time.time() - start_time

        mlb = MultiLabelBinarizer(classes=test_label_universe)
        y_true_bin = mlb.fit_transform(gold_labels)
        y_pred_bin = mlb.transform(pred_labels)

        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="macro", zero_division=0
        )

        acc1 = flat_acc_at_1(gold_labels, pred_labels)
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
                "acc_at_1": float(acc1),
            },
            "hierarchical_levels": level_metrics,
            "fewshot": {
                "static_k": int(args.static_k),
                "dynamic_k": int(args.dynamic_k),
                "retriever": "e5-base-v2",
                "static_example_ids": static_ids,
                "dynamic_retrieval": "semantic_similarity_over_train_embeddings",
                "decoding": {"do_sample": False},
                "max_labels_per_pred": int(args.max_labels),
            },
        }
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for idx, (t, a, gold, pred, rawtxt, prompt, meta) in enumerate(
                zip(test_titles, test_abstracts, gold_labels, pred_labels, raw_generations, prompts_all, retrieval_meta_all)
            ):
                rec = {
                    "idx": idx,
                    "title": t,
                    "abstract": a,
                    "gold_labels": gold,
                    "pred_labels": pred,
                    "raw_generation": rawtxt,
                    "prompt": prompt,
                    "retrieval": meta,
                }
                f.write(json.dumps(rec) + "\n")

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
