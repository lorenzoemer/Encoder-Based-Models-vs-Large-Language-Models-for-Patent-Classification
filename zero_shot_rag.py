#!/usr/bin/env python3
"""
Zero-shot + allowed-set retrieval (RAG-style) evaluation of LLMs for CPC multi-label patent classification.

ZERO-SHOT + RAG setting:
- No few-shot examples
- Allowed-label set is retrieved per patent from CPC subclass definitions
- Retriever: E5-base-v2 (bi-encoder), Top-K = 20
- Deterministic decoding (do_sample=False)
- Output constraint: 1 to 7 CPC subclasses (4-character level)
- Output restriction: predictions must be a subset of the retrieved allowed set
- Parsing: strict JSON {"labels": [...]} with a fallback to extracting the first JSON object
- Non-empty: if parsing yields empty, fall back to first allowed label (Top-1 retrieved)

Inputs:
- TSV test file with columns: title, abstract, labels
- CPC definitions file (JSON): {"A01B": "...", "G06F": "...", ...}

Outputs (per model) under: <output_dir>/<model_name>/
- metrics.json
- predictions.jsonl (includes allowed_topk and prompt)
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
    "You MUST only choose labels from the provided allowed set. "
    "If a CPC subclass is plausibly relevant, you SHOULD include it. "
    "Return ONLY a strict JSON object with key \"labels\" mapping to a list of CPC subclass codes. "
    "Return JSON ONLY."
)

USER_PROMPT_TEMPLATE = (
    "NOW CLASSIFY THIS PATENT:\n"
    "----------------------------------------\n"
    "{patent_text}\n"
    "----------------------------------------\n\n"
    "ALLOWED CPC SUBCLASSES (retrieved; with short definitions):\n"
    "{allowed_labels_block}\n\n"
    "TASK:\n"
    "- Choose ALL relevant CPC subclasses from the allowed list.\n"
    "- Prefer OVER-INCLUSION to under-inclusion.\n"
    "- Output between 1 and {max_labels} CPC subclass codes.\n\n"
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


def parse_predicted_labels(text: str, allowed_set: Optional[set], max_labels: int) -> List[str]:
    if not text:
        return []

    def postprocess(labs: List[str]) -> List[str]:
        norm = [normalize_cpc_label(x) for x in labs]
        norm = [x for x in norm if x]
        if allowed_set is not None:
            norm = [x for x in norm if x in allowed_set]
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


def enforce_nonempty(pred: List[str], raw_text: str, allowed_list: List[str], max_labels: int) -> List[str]:
    if pred:
        return pred[:max_labels]

    allowed_set = set(allowed_list or [])
    cands = _CPC_TOKEN_RE.findall((raw_text or "").upper())
    for c in cands:
        c = normalize_cpc_label(c)
        if c and (not allowed_set or c in allowed_set):
            return [c]

    if allowed_list:
        return [normalize_cpc_label(allowed_list[0])]

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


def group_allowed_labels_hierarchically(allowed_labels: List[str], cpc_desc_map: Dict[str, str]) -> str:
    from collections import defaultdict

    tree = defaultdict(lambda: defaultdict(list))
    for raw_code in allowed_labels:
        code = normalize_cpc_label(raw_code)
        if not code:
            continue
        section = code[0]
        cclass = code[:3]
        tree[section][cclass].append(code)

    lines: List[str] = []
    for section in sorted(tree.keys()):
        lines.append(f"Section {section}:")
        for cclass in sorted(tree[section].keys()):
            lines.append(f"  Class {cclass}:")
            for code in sorted(set(tree[section][cclass])):
                desc = (cpc_desc_map.get(code, "") or "").strip()
                if len(desc) > 220:
                    desc = desc[:220].rstrip() + "…"
                lines.append(f"    - {code}" + (f" — {desc}" if desc else ""))
        lines.append("")
    return "\n".join(lines).strip()


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
def retrieve_allowed_topk(
    patent_texts: List[str],
    k: int,
    embed_model: AutoModel,
    embed_tokenizer: AutoTokenizer,
    embed_device: str,
    label_embeds_cpu: torch.Tensor,
    label_codes: List[str],
    patent_emb_batch: int,
) -> List[List[str]]:
    q_emb_cpu = encode_texts_e5(
        patent_texts,
        embed_model,
        embed_tokenizer,
        embed_device,
        batch_size=min(patent_emb_batch, max(1, len(patent_texts))),
        prefix="query",
    )
    q = q_emb_cpu.to(embed_device)
    L = label_embeds_cpu.to(embed_device)
    sims = q @ L.T
    top_idx = torch.topk(sims, k=min(k, L.size(0)), dim=1).indices.cpu().tolist()
    return [[label_codes[j] for j in idxs] for idxs in top_idx]


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
    allowed_lists: List[List[str]],
    cpc_defs: Dict[str, str],
    max_new_tokens: int,
    max_labels: int,
) -> Tuple[List[List[str]], List[str], List[str], List[Dict[str, Any]]]:
    prompts: List[str] = []
    retrieval_meta: List[Dict[str, Any]] = []

    for title, abstract, allowed in zip(titles, abstracts, allowed_lists):
        qtext = build_text(title, abstract)
        allowed_norm = [normalize_cpc_label(x) for x in (allowed or [])]
        allowed_norm = [x for x in allowed_norm if x]
        allowed_norm = _dedupe_keep_order(allowed_norm)

        allowed_block = group_allowed_labels_hierarchically(allowed_norm, cpc_defs) if allowed_norm else "(none)"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            patent_text=qtext,
            allowed_labels_block=allowed_block,
            max_labels=max_labels,
        )
        full_prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)

        prompts.append(full_prompt)
        retrieval_meta.append({"allowed_topk": allowed_norm})

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

        allowed_here = retrieval_meta[i]["allowed_topk"]
        allowed_set = set(allowed_here)

        labs = parse_predicted_labels(gen_text, allowed_set=allowed_set, max_labels=max_labels)
        labs = enforce_nonempty(labs, gen_text, allowed_here, max_labels=max_labels)

        clean = [normalize_cpc_label(x) for x in labs]
        clean = [x for x in clean if x in allowed_set]
        clean = _dedupe_keep_order(clean)[:max_labels]
        preds.append(clean)

    return preds, raw_texts, prompts, retrieval_meta


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
    embed_model: AutoModel,
    embed_tokenizer: AutoTokenizer,
    embed_device: str,
    label_codes: List[str],
    label_embeds_cpu: torch.Tensor,
    cpc_defs: Dict[str, str],
    top_k: int,
    patent_emb_batch: int,
    local_only: bool,
    use_codecarbon: bool,
) -> None:
    outdir = out_root / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    test_ds = load_dataset("csv", data_files={"test": str(test_path)}, delimiter="\t")["test"]
    if max_eval is not None:
        test_ds = test_ds.select(range(min(int(max_eval), len(test_ds))))

    titles: List[str] = []
    abstracts: List[str] = []
    patent_texts: List[str] = []
    gold_labels: List[List[str]] = []

    for ex in test_ds:
        t = ex.get(title_col, "") or ""
        a = ex.get(abstract_col, "") or ""
        titles.append(t)
        abstracts.append(a)
        patent_texts.append(build_text(t, a))

        labs = [normalize_cpc_label(x) for x in parse_labels(ex.get(labels_col))]
        labs = [x for x in labs if x]
        gold_labels.append(labs)

    test_label_universe = sorted(set(l for labs in gold_labels for l in labs))
    if not test_label_universe:
        raise ValueError("No gold labels found after parsing test labels.")

    model, tok = load_llm(model_path, local_only=local_only)

    tracker = None
    if use_codecarbon and EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name=f"zero_shot_rag_{model_name}_topk{top_k}",
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
    retrieval_meta_all: List[Dict[str, Any]] = []

    for i in tqdm(range(0, len(patent_texts), batch_size), desc=f"Inference [{model_name}]"):
        btxt = patent_texts[i : i + batch_size]
        bt = titles[i : i + batch_size]
        ba = abstracts[i : i + batch_size]

        allowed_lists = retrieve_allowed_topk(
            patent_texts=btxt,
            k=top_k,
            embed_model=embed_model,
            embed_tokenizer=embed_tokenizer,
            embed_device=embed_device,
            label_embeds_cpu=label_embeds_cpu,
            label_codes=label_codes,
            patent_emb_batch=patent_emb_batch,
        )

        batch_preds, batch_raw, batch_prompts, batch_meta = generate_batch(
            model=model,
            tokenizer=tok,
            titles=bt,
            abstracts=ba,
            allowed_lists=allowed_lists,
            cpc_defs=cpc_defs,
            max_new_tokens=max_new_tokens,
            max_labels=max_labels,
        )

        pred_labels.extend(batch_preds)
        raw_generations.extend(batch_raw)
        prompts_all.extend(batch_prompts)
        retrieval_meta_all.extend(batch_meta)

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
        "rag": {
            "retriever": "e5-base-v2",
            "top_k": int(top_k),
            "allowed_labels_source": "cpc_definitions",
            "restrict_to_allowed_set": True,
            "max_labels_per_pred": int(max_labels),
            "decoding": {"do_sample": False},
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, (t, a, ptxt, gold, pred, rawtxt, prompt, meta) in enumerate(
            zip(
                titles,
                abstracts,
                patent_texts,
                gold_labels,
                pred_labels,
                raw_generations,
                prompts_all,
                retrieval_meta_all,
            )
        ):
            rec = {
                "idx": idx,
                "title": t,
                "abstract": a,
                "patent_text": ptxt,
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


def load_cpc_definitions(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("CPC definitions file must be a JSON object mapping code -> definition.")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        code = normalize_cpc_label(k)
        if not code:
            continue
        out[code] = str(v) if v is not None else ""
    if not out:
        raise ValueError("No valid CPC codes found in definitions file.")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_path", type=Path, required=True)
    ap.add_argument("--cpc_definitions_path", type=Path, required=True)

    ap.add_argument("--embed_model_path", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--title_col", type=str, default="title")
    ap.add_argument("--abstract_col", type=str, default="abstract")
    ap.add_argument("--labels_col", type=str, default="labels")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_eval", type=int, default=None)

    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_labels", type=int, default=7)

    ap.add_argument("--cpc_emb_batch", type=int, default=64)
    ap.add_argument("--patent_emb_batch", type=int, default=16)

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
    if not args.cpc_definitions_path.is_file():
        raise FileNotFoundError(f"Missing CPC definitions file: {args.cpc_definitions_path}")
    if not args.embed_model_path.is_dir():
        raise FileNotFoundError(f"Missing embedding model directory: {args.embed_model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpc_defs = load_cpc_definitions(args.cpc_definitions_path)

    embed_model, embed_tok, embed_dev = load_encoder(args.embed_model_path, local_only=args.local_only)

    label_codes = sorted(cpc_defs.keys())
    label_texts = [f"{code}: {(cpc_defs.get(code, '') or '').strip()}" for code in label_codes]
    label_embeds_cpu = encode_texts_e5(
        texts=label_texts,
        model=embed_model,
        tokenizer=embed_tok,
        device=embed_dev,
        batch_size=args.cpc_emb_batch,
        prefix="passage",
    ).cpu()

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
            embed_model=embed_model,
            embed_tokenizer=embed_tok,
            embed_device=embed_dev,
            label_codes=label_codes,
            label_embeds_cpu=label_embeds_cpu,
            cpc_defs=cpc_defs,
            top_k=args.top_k,
            patent_emb_batch=args.patent_emb_batch,
            local_only=args.local_only,
            use_codecarbon=args.use_codecarbon,
        )


if __name__ == "__main__":
    main()
