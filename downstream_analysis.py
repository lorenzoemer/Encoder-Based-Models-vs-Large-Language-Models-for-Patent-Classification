#!/usr/bin/env python3
"""
Patentometric analysis of downstream consequences of different CPC classification models.

The script measures section- and subclass-level F1, routed-subset gains,
semantic breadth, normalized absolute count error, and country/assignee
rank displacement relative to gold CPC assignments.
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


YEAR_COLUMNS = ["grant_year", "filing_year", "year", "publication_year", "priority_year"]
COUNTRY_COLUMNS = ["primary_assignee_country", "assignee_country", "applicant_country", "country"]
ASSIGNEE_COLUMNS = ["primary_assignee_name", "primary_assignee_id", "assignee", "applicant", "organisation", "organization"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc evaluation of CPC classification predictions.")
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--encoder-predictions", type=Path, required=True)
    parser.add_argument("--qwen-predictions", type=Path)
    parser.add_argument("--hybrid", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--embedding-model", type=str)
    parser.add_argument("--embedding-tokenizer", type=str)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--wilcoxon-min-support", type=int, default=1)
    parser.add_argument("--min-entity-patents", type=int, default=10)
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser


def normalize_cpc(value: Any) -> str:
    code = re.sub(r"[^A-Z0-9]", "", str(value or "").upper().split("/")[0])
    if len(code) >= 4 and code[0].isalpha() and code[1:3].isdigit() and code[3].isalpha():
        return code[:4]
    return ""


def dedupe(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        code = normalize_cpc(value)
        if code and code not in seen:
            seen.add(code)
            output.append(code)
    return output


def parse_labels(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def read_tsv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "patent_id" not in frame.columns:
        if "id" in frame.columns:
            frame["patent_id"] = frame["id"].astype(str)
        else:
            frame["patent_id"] = np.arange(len(frame)).astype(str)
    frame["patent_id"] = frame["patent_id"].astype(str)
    frame["gold"] = frame["labels"].map(parse_labels)
    frame["text"] = (
        frame["title"].fillna("").str.strip()
        + ". "
        + frame["abstract"].fillna("").str.strip()
    ).str.strip(". ")
    return frame


def read_prediction_file(path: Path, default_prediction_field: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            obj = json.loads(line)
            patent_id = str(obj.get("patent_id", obj.get("id", obj.get("idx", obj.get("index", row_number)))))
            prediction = obj.get(default_prediction_field)
            if prediction is None:
                for key in ("pred", "predicted", "prediction", "hybrid_pred", "qwen_pred"):
                    if key in obj:
                        prediction = obj[key]
                        break
            rows.append(
                {
                    "patent_id": patent_id,
                    "pred": parse_labels(prediction),
                    "encoder_pred": parse_labels(obj.get("encoder_pred", [])),
                    "gold_from_file": parse_labels(obj.get("gold", obj.get("gold_labels", []))),
                    "routed": parse_bool(obj.get("routed", obj.get("routed_to_llm", False))),
                    "routing_fraction": obj.get("routing_fraction"),
                    "uncertainty": obj.get("uncertainty_score", obj.get("uncertainty")),
                }
            )
    return pd.DataFrame(rows)


def parse_hybrid_arguments(values: Sequence[str]) -> List[Tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --hybrid value: {value}. Use NAME=PATH.")
        name, path = value.split("=", 1)
        output.append((name.strip(), Path(path.strip())))
    return output


def align_predictions(test: pd.DataFrame, predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if len(predictions) != len(test):
        raise RuntimeError(f"{model_name}: {len(predictions)} predictions for {len(test)} test patents")

    test_ids = set(test["patent_id"])
    prediction_ids = set(predictions["patent_id"])

    if test_ids == prediction_ids:
        merged = test.merge(predictions, on="patent_id", how="left", validate="one_to_one")
    else:
        merged = test.reset_index(drop=True).copy()
        aligned = predictions.reset_index(drop=True)
        for column in aligned.columns:
            if column != "patent_id":
                merged[column] = aligned[column]

    if merged["pred"].isna().any():
        raise RuntimeError(f"{model_name}: missing aligned predictions")

    return merged


def load_models(args: argparse.Namespace, test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    models: Dict[str, pd.DataFrame] = {}

    encoder = read_prediction_file(args.encoder_predictions, "pred")
    models["Encoder"] = align_predictions(test, encoder, "Encoder")

    if args.qwen_predictions:
        qwen = read_prediction_file(args.qwen_predictions, "pred")
        models["Qwen-LoRA"] = align_predictions(test, qwen, "Qwen-LoRA")

    for name, path in parse_hybrid_arguments(args.hybrid):
        hybrid = read_prediction_file(path, "hybrid_pred")
        aligned = align_predictions(test, hybrid, name)
        if aligned["encoder_pred"].map(len).eq(0).all():
            aligned["encoder_pred"] = models["Encoder"]["pred"]
        models[name] = aligned

    return models


def collapse(labels: Sequence[str], level: str) -> List[str]:
    if level == "section":
        return sorted({label[0] for label in labels})
    if level == "class":
        return sorted({label[:3] for label in labels})
    return sorted(set(labels))


def per_label_metrics(frame: pd.DataFrame, model: str, level: str) -> pd.DataFrame:
    gold = [collapse(labels, level) for labels in frame["gold"]]
    pred = [collapse(labels, level) for labels in frame["pred"]]
    labels = sorted({label for row in gold + pred for label in row})
    binarizer = MultiLabelBinarizer(classes=labels)
    y_true = binarizer.fit_transform(gold)
    y_pred = binarizer.transform(pred)

    rows = []
    for index, label in enumerate(labels):
        true = y_true[:, index]
        predicted = y_pred[:, index]
        tp = int(((true == 1) & (predicted == 1)).sum())
        fp = int(((true == 0) & (predicted == 1)).sum())
        fn = int(((true == 1) & (predicted == 0)).sum())
        tn = int(((true == 0) & (predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "model": model,
                "level": level,
                "label": label,
                "section": label[0],
                "support": int(true.sum()),
                "predicted_positive": int(predicted.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_bias": int(predicted.sum() - true.sum()),
            }
        )
    return pd.DataFrame(rows)


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def paired_label_f1(
    gold_rows: Sequence[Sequence[str]],
    encoder_rows: Sequence[Sequence[str]],
    hybrid_rows: Sequence[Sequence[str]],
    level: str,
    min_support: int,
) -> pd.DataFrame:
    gold = [collapse(row, level) for row in gold_rows]
    encoder = [collapse(row, level) for row in encoder_rows]
    hybrid = [collapse(row, level) for row in hybrid_rows]
    labels = sorted({label for rows in (gold, encoder, hybrid) for row in rows for label in row})

    rows = []
    for label in labels:
        y_true = np.fromiter((label in row for row in gold), dtype=np.int8)
        if int(y_true.sum()) < min_support:
            continue
        y_encoder = np.fromiter((label in row for row in encoder), dtype=np.int8)
        y_hybrid = np.fromiter((label in row for row in hybrid), dtype=np.int8)
        encoder_f1 = binary_f1(y_true, y_encoder)
        hybrid_f1 = binary_f1(y_true, y_hybrid)
        rows.append(
            {
                "level": level,
                "label": label,
                "section": label[0],
                "support": int(y_true.sum()),
                "encoder_f1": encoder_f1,
                "hybrid_f1": hybrid_f1,
                "delta_f1": hybrid_f1 - encoder_f1,
            }
        )
    return pd.DataFrame(rows)


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def safe_wilcoxon(differences: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0):
        return 0.0, 1.0
    result = wilcoxon(values, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
    return float(result.statistic), float(result.pvalue)


def bonferroni(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric * max(int(numeric.notna().sum()), 1)).clip(upper=1.0)


def hybrid_wilcoxon(models: Dict[str, pd.DataFrame], min_support: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    encoder = models["Encoder"]
    tests = []
    details = []

    for name, hybrid in models.items():
        if name in {"Encoder", "Qwen-LoRA"}:
            continue

        routed = hybrid["routed"].fillna(False).to_numpy(dtype=bool)
        samples = {
            "full_test_set": np.ones(len(hybrid), dtype=bool),
            "routed_patents_only": routed,
        }

        for sample_name, mask in samples.items():
            if int(mask.sum()) == 0:
                continue

            gold_rows = hybrid.loc[mask, "gold"].tolist()
            encoder_rows = encoder.loc[mask, "pred"].tolist()
            hybrid_rows = hybrid.loc[mask, "pred"].tolist()

            for level in ("section", "class", "subclass"):
                paired = paired_label_f1(gold_rows, encoder_rows, hybrid_rows, level, min_support)
                if paired.empty:
                    continue
                differences = paired["delta_f1"].to_numpy(dtype=float)
                statistic, p_value = safe_wilcoxon(differences)
                tests.append(
                    {
                        "model": name,
                        "evaluation_sample": sample_name,
                        "level": level,
                        "n_patents": int(mask.sum()),
                        "n_labels": len(paired),
                        "improved_labels": int((differences > 0).sum()),
                        "worsened_labels": int((differences < 0).sum()),
                        "tied_labels": int((differences == 0).sum()),
                        "mean_delta_f1": float(differences.mean()),
                        "median_delta_f1": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "p_value_raw": p_value,
                        "rank_biserial_r": rank_biserial(differences),
                    }
                )
                paired.insert(0, "model", name)
                paired.insert(1, "evaluation_sample", sample_name)
                details.append(paired)

    tests_frame = pd.DataFrame(tests)
    details_frame = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    if not tests_frame.empty:
        tests_frame["p_value_bonferroni"] = np.nan
        for _, indices in tests_frame.groupby(["evaluation_sample", "level"]).groups.items():
            tests_frame.loc[indices, "p_value_bonferroni"] = bonferroni(tests_frame.loc[indices, "p_value_raw"])
        tests_frame["significant_0_05"] = tests_frame["p_value_bonferroni"] < 0.05

    return tests_frame, details_frame


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.inference_mode()
def compute_embeddings(
    texts: Sequence[str],
    model_name: str,
    tokenizer_name: Optional[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        pooled = mean_pool(hidden, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        output.append(pooled.cpu().numpy())

    return np.concatenate(output, axis=0)


def semantic_breadth(embeddings: np.ndarray, gold: Sequence[Sequence[str]], min_support: int) -> pd.DataFrame:
    rows = []
    labels = sorted({label for row in gold for label in row})

    for label in labels:
        indices = np.array([index for index, row in enumerate(gold) if label in row], dtype=int)
        if len(indices) < max(min_support, 2):
            continue
        values = normalize(embeddings[indices].astype(np.float64))
        total = values.sum(axis=0, keepdims=True)
        centroids = normalize((total - values) / (len(values) - 1))
        similarities = np.sum(values * centroids, axis=1)
        distances = 1.0 - similarities
        rows.append(
            {
                "label": label,
                "section": label[0],
                "support": len(indices),
                "semantic_breadth": float(distances.mean()),
                "semantic_breadth_median": float(np.median(distances)),
                "semantic_breadth_sd": float(distances.std(ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def breadth_analysis(
    breadth: pd.DataFrame,
    subclass_metrics: pd.DataFrame,
    train_frequency: Counter,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = subclass_metrics.merge(
        breadth.rename(columns={"support": "breadth_support"}),
        on=["label", "section"],
        how="inner",
        validate="many_to_one",
    )
    merged["test_support"] = merged["support"].astype(float)
    merged["train_frequency"] = merged["label"].map(train_frequency).fillna(0).astype(float)

    correlations = []
    for model, group in merged.groupby("model"):
        for variable in ("semantic_breadth", "train_frequency", "test_support"):
            valid = group[[variable, "f1"]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p_value = spearmanr(valid[variable], valid["f1"]) if len(valid) >= 5 else (np.nan, np.nan)
            correlations.append(
                {
                    "model": model,
                    "variable": variable,
                    "n": len(valid),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    correlation_frame = pd.DataFrame(correlations)
    regression_rows = []

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests

        merged["log_train_frequency"] = np.log1p(merged["train_frequency"])
        merged["log_test_support"] = np.log1p(merged["test_support"])

        for model, group in merged.groupby("model"):
            sample = group.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["f1", "semantic_breadth", "log_train_frequency", "log_test_support", "section"]
            )
            if len(sample) < 20:
                continue
            fit = smf.ols(
                "f1 ~ semantic_breadth + log_train_frequency + log_test_support + C(section)",
                data=sample,
            ).fit(cov_type="HC3")
            for term in fit.params.index:
                regression_rows.append(
                    {
                        "model": model,
                        "term": term,
                        "coefficient": fit.params[term],
                        "std_error": fit.bse[term],
                        "p_value": fit.pvalues[term],
                        "r_squared": fit.rsquared,
                        "n": int(fit.nobs),
                    }
                )

        valid = correlation_frame["p_value"].notna()
        if valid.any():
            correlation_frame.loc[valid, "p_value_fdr"] = multipletests(
                correlation_frame.loc[valid, "p_value"], method="fdr_bh"
            )[1]

        regression_frame = pd.DataFrame(regression_rows)
        if not regression_frame.empty:
            valid = regression_frame["p_value"].notna()
            regression_frame.loc[valid, "p_value_fdr"] = multipletests(
                regression_frame.loc[valid, "p_value"], method="fdr_bh"
            )[1]
    except ImportError:
        regression_frame = pd.DataFrame(regression_rows)

    return merged, correlation_frame, regression_frame


def first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((column for column in candidates if column in frame.columns), None)


def explode_assignments(
    frame: pd.DataFrame,
    label_column: str,
    level: str,
    entity: Optional[str] = None,
) -> pd.DataFrame:
    columns = ["patent_id", label_column] + ([entity] if entity else [])
    exploded = frame[columns].copy()
    exploded["technology"] = exploded[label_column].map(lambda labels: collapse(labels, level))
    exploded = exploded.explode("technology")
    exploded = exploded[exploded["technology"].notna() & exploded["technology"].ne("")]
    if entity:
        exploded = exploded[exploded[entity].astype(str).str.strip().ne("")]
    return exploded


def count_distortion(frame: pd.DataFrame, model: str, level: str) -> pd.DataFrame:
    gold = explode_assignments(frame, "gold", level).groupby("technology")["patent_id"].nunique().rename("gold_count")
    pred = explode_assignments(frame, "pred", level).groupby("technology")["patent_id"].nunique().rename("pred_count")
    output = pd.concat([gold, pred], axis=1).fillna(0).reset_index()
    output["model"] = model
    output["level"] = level
    output["absolute_count_error"] = (output["pred_count"] - output["gold_count"]).abs()
    output["signed_count_error"] = output["pred_count"] - output["gold_count"]
    return output


def count_error_summary(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level), group in counts.groupby(["model", "level"]):
        total_gold = float(group["gold_count"].sum())
        overall = float(group["absolute_count_error"].sum() / total_gold) if total_gold else np.nan
        y_group = group[group["technology"].astype(str).str.startswith("Y")]
        y_gold = float(y_group["gold_count"].sum())
        y_error = float(y_group["absolute_count_error"].sum() / y_gold) if y_gold else np.nan
        rows.append(
            {
                "model": model,
                "level": level,
                "scope": "complete_taxonomy",
                "normalized_absolute_count_error": overall,
            }
        )
        rows.append(
            {
                "model": model,
                "level": level,
                "scope": "section_y",
                "normalized_absolute_count_error": y_error,
            }
        )
    return pd.DataFrame(rows)


def ranking_distortion(
    frame: pd.DataFrame,
    model: str,
    level: str,
    entity: str,
    min_entity_patents: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entity_counts = frame.groupby(entity)["patent_id"].nunique()
    eligible = set(entity_counts[entity_counts >= min_entity_patents].index.astype(str))

    def panel(label_column: str, count_name: str) -> pd.Series:
        exploded = explode_assignments(frame, label_column, level, entity)
        exploded = exploded[exploded[entity].astype(str).isin(eligible)]
        return exploded.groupby([entity, "technology"])["patent_id"].nunique().rename(count_name)

    combined = pd.concat([panel("gold", "gold_count"), panel("pred", "pred_count")], axis=1).fillna(0).reset_index()
    detail_rows = []
    summary_rows = []

    for technology, group in combined.groupby("technology"):
        if len(group) < 3:
            continue
        gold_rank = group["gold_count"].rank(method="min", ascending=False)
        pred_rank = group["pred_count"].rank(method="min", ascending=False)
        displacement = (pred_rank - gold_rank).abs()
        rho, p_value = spearmanr(group["gold_count"], group["pred_count"])
        for row_index, (_, row) in enumerate(group.iterrows()):
            detail_rows.append(
                {
                    "model": model,
                    "level": level,
                    "entity_column": entity,
                    "technology": technology,
                    "entity": row[entity],
                    "gold_count": row["gold_count"],
                    "pred_count": row["pred_count"],
                    "gold_rank": float(gold_rank.iloc[row_index]),
                    "pred_rank": float(pred_rank.iloc[row_index]),
                    "absolute_rank_displacement": float(displacement.iloc[row_index]),
                }
            )
        summary_rows.append(
            {
                "model": model,
                "level": level,
                "entity_column": entity,
                "technology": technology,
                "section": str(technology)[0],
                "n_entities": len(group),
                "mean_absolute_rank_displacement": float(displacement.mean()),
                "max_absolute_rank_displacement": float(displacement.max()),
                "spearman_rho": rho,
                "spearman_p": p_value,
            }
        )

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def ranking_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level, entity), group in rankings.groupby(["model", "level", "entity_column"]):
        rows.append(
            {
                "model": model,
                "level": level,
                "entity_column": entity,
                "scope": "complete_taxonomy",
                "mean_absolute_rank_displacement": float(group["mean_absolute_rank_displacement"].mean()),
                "n_technologies": len(group),
            }
        )
        y_group = group[group["section"] == "Y"]
        rows.append(
            {
                "model": model,
                "level": level,
                "entity_column": entity,
                "scope": "section_y",
                "mean_absolute_rank_displacement": float(y_group["mean_absolute_rank_displacement"].mean()) if len(y_group) else np.nan,
                "n_technologies": len(y_group),
            }
        )
    return pd.DataFrame(rows)


def save(frame: Optional[pd.DataFrame], path: Path) -> None:
    if frame is not None and not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test = read_tsv(args.test_path)
    train = read_tsv(args.train_path)
    train_frequency = Counter(label for row in train["gold"] for label in row)
    models = load_models(args, test)

    metrics = []
    for model, frame in models.items():
        for level in ("section", "class", "subclass"):
            metrics.append(per_label_metrics(frame, model, level))
    metrics_frame = pd.concat(metrics, ignore_index=True)
    save(metrics_frame, args.out_dir / "performance_all_levels.csv")
    save(metrics_frame[metrics_frame["level"] == "section"], args.out_dir / "performance_sections.csv")
    save(metrics_frame[metrics_frame["level"] == "subclass"], args.out_dir / "performance_subclasses.csv")
    save(
        metrics_frame[(metrics_frame["level"] == "section") & (metrics_frame["label"] == "Y")],
        args.out_dir / "section_y_performance.csv",
    )

    wilcoxon_tests, wilcoxon_details = hybrid_wilcoxon(models, args.wilcoxon_min_support)
    save(wilcoxon_tests, args.out_dir / "hybrid_wilcoxon_tests.csv")
    save(wilcoxon_details, args.out_dir / "hybrid_wilcoxon_per_label.csv")

    if not args.skip_embeddings:
        if not args.embedding_model:
            raise ValueError("--embedding-model is required unless --skip-embeddings is used")
        if args.embedding_cache and args.embedding_cache.is_file():
            embeddings = np.load(args.embedding_cache)
            if len(embeddings) != len(test):
                raise RuntimeError("Embedding cache length does not match the test set")
        else:
            embeddings = compute_embeddings(
                test["text"].tolist(),
                args.embedding_model,
                args.embedding_tokenizer,
                args.batch_size,
                args.max_length,
            )
            if args.embedding_cache:
                args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.embedding_cache, embeddings)

        breadth = semantic_breadth(embeddings, test["gold"].tolist(), args.min_support)
        subclass_metrics = metrics_frame[metrics_frame["level"] == "subclass"]
        merged, correlations, regressions = breadth_analysis(breadth, subclass_metrics, train_frequency)
        save(breadth, args.out_dir / "semantic_breadth_subclasses.csv")
        save(merged, args.out_dir / "semantic_breadth_performance.csv")
        save(correlations, args.out_dir / "semantic_breadth_correlations.csv")
        save(regressions, args.out_dir / "semantic_breadth_regressions.csv")

    count_frames = []
    ranking_details = []
    ranking_summaries = []

    country_column = first_column(test, COUNTRY_COLUMNS)
    assignee_column = first_column(test, ASSIGNEE_COLUMNS)

    for model, frame in models.items():
        for level in ("section", "class", "subclass"):
            count_frames.append(count_distortion(frame, model, level))
            for entity in (country_column, assignee_column):
                if not entity:
                    continue
                details, summaries = ranking_distortion(
                    frame,
                    model,
                    level,
                    entity,
                    args.min_entity_patents,
                )
                if not details.empty:
                    ranking_details.append(details)
                if not summaries.empty:
                    ranking_summaries.append(summaries)

    counts = pd.concat(count_frames, ignore_index=True)
    save(counts, args.out_dir / "downstream_count_distortion_by_technology.csv")
    save(count_error_summary(counts), args.out_dir / "downstream_count_distortion_summary.csv")

    if ranking_details:
        details = pd.concat(ranking_details, ignore_index=True)
        save(details, args.out_dir / "downstream_rank_displacement_by_entity.csv")

    if ranking_summaries:
        summaries = pd.concat(ranking_summaries, ignore_index=True)
        save(summaries, args.out_dir / "downstream_rank_displacement_by_technology.csv")
        save(ranking_summary(summaries), args.out_dir / "downstream_rank_displacement_summary.csv")


if __name__ == "__main__":
    main()
