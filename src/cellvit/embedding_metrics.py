"""Metadata-aware, bounded-memory evaluation of well-level representations.

Only NumPy and PyArrow are required. No preprocessing is fitted in this module:
fit scalers/PCA on training wells before calling these functions. An evaluation
is meaningful only when every model uses the same aligned wells and protocol.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EmbeddingTable:
    well_ids: tuple[str, ...]
    values: np.ndarray
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or len(self.well_ids) != len(self.values):
            raise ValueError("Embedding rows must match well_ids and be two-dimensional")
        if not self.well_ids or self.values.shape[1] == 0:
            raise ValueError("Embedding table is empty")
        if any(not isinstance(key, str) or not key for key in self.well_ids):
            raise ValueError("well_id must be a nonempty string")
        if len(set(self.well_ids)) != len(self.well_ids):
            raise ValueError("Duplicate embedding well_id")


def load_embedding_parquet(
    path: Path | str, *, well_ids: Iterable[str] | None = None, batch_size: int = 2048,
) -> EmbeddingTable:
    """Read list-vector exports or the public wide-column RxRx3 embeddings.

    Every ID is scanned and duplicate IDs fail even outside the selected subset.
    Only selected feature rows are retained in memory. CellProfiler's pandas
    index is excluded; NaN/Inf are left for explicit alignment policy below.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if "well_id" not in schema.names:
        raise ValueError(f"Missing well_id in {path}")
    vector_column = "embedding" if "embedding" in schema.names else None
    if vector_column:
        dtype = schema.field(vector_column).type
        if not (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)
                or pa.types.is_large_list(dtype)):
            raise ValueError("embedding must contain list vectors")
        features = [vector_column]
    else:
        numbered = [name for name in schema.names
                    if name.startswith("feature_") and name[8:].isdigit()]
        if numbered:
            features = sorted(numbered, key=lambda name: int(name[8:]))
        else:
            excluded = {"well_id", "plate", "concentration", "site", "channel"}
            features = [field.name for field in schema
                        if field.name not in excluded and not field.name.startswith("__index_")
                        and (pa.types.is_floating(field.type) or pa.types.is_integer(field.type))]
    if not features:
        raise ValueError(f"No numeric features found in {path}")
    wanted = None if well_ids is None else set(well_ids)
    if wanted is not None and not wanted:
        raise ValueError("well_ids selection is empty")
    seen: set[str] = set()
    retained: list[str] = []
    chunks: list[np.ndarray] = []
    width: int | None = None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["well_id", *features]):
        ids = batch.column(0).to_pylist()
        for key in ids:
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError(f"Missing or duplicate embedding well_id: {key!r}")
            seen.add(key)
        positions = [i for i, key in enumerate(ids) if wanted is None or key in wanted]
        if not positions:
            continue
        selected = batch.take(pa.array(positions, type=pa.int64()))
        if vector_column:
            try:
                matrix = np.asarray(selected.column(1).to_pylist(), dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError("Embedding vectors must have a consistent numeric width") from error
            if matrix.ndim != 2 or matrix.shape[1] == 0:
                raise ValueError("Embedding vectors must have a consistent nonzero width")
        else:
            matrix = np.column_stack([
                selected.column(i).to_numpy(zero_copy_only=False)
                for i in range(1, selected.num_columns)
            ]).astype(np.float32, copy=False)
        if width is not None and matrix.shape[1] != width:
            raise ValueError("Embedding vector width changes between batches")
        width = matrix.shape[1]
        retained.extend(ids[i] for i in positions)
        chunks.append(matrix)
    if not retained:
        raise ValueError(f"No requested embedding wells found in {path}")
    return EmbeddingTable(tuple(retained), np.concatenate(chunks), tuple(features))


def align_embedding_tables(
    tables: Mapping[str, EmbeddingTable], metadata: Mapping[str, Mapping[str, str]], *,
    well_ids: Iterable[str] | None = None, drop_nonfinite: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict]:
    """Strict one-to-one ID joins followed by a common candidate intersection.

    Missing wells are reported, never aligned by row position. Unknown IDs and
    malformed metadata fail. If requested, nonfinite rows are removed from ALL
    models together and the removals are reported.
    """
    if not tables or not metadata:
        raise ValueError("Need at least one model and nonempty metadata")
    for key, row in metadata.items():
        if not key or row.get("well_id") != key:
            raise ValueError("Metadata keys and well_id values disagree")
    selected = set(metadata) if well_ids is None else set(well_ids)
    if not selected or not selected <= set(metadata):
        raise ValueError("Selected wells must be a nonempty metadata subset")
    common = selected.copy()
    model_report = {}
    for name, table in tables.items():
        keys = set(table.well_ids)
        unknown = keys - set(metadata)
        if unknown:
            raise ValueError(f"{name}: embedding IDs absent from metadata: {sorted(unknown)[:3]}")
        common &= keys
        model_report[name] = {
            "n_loaded": len(keys), "n_selected_present": len(keys & selected),
            "n_selected_missing": len(selected - keys), "dimension": table.values.shape[1],
        }
    if not common:
        raise ValueError("No common wells across models and metadata")
    ordered = sorted(common)
    aligned = {}
    finite = np.ones(len(ordered), dtype=bool)
    for name, table in tables.items():
        index = {key: i for i, key in enumerate(table.well_ids)}
        values = table.values[[index[key] for key in ordered]].astype(np.float32, copy=False)
        good = np.isfinite(values).all(axis=1)
        model_report[name]["n_nonfinite_common"] = int((~good).sum())
        finite &= good
        aligned[name] = values
    dropped = [key for key, good in zip(ordered, finite) if not good]
    if dropped and not drop_nonfinite:
        raise ValueError(f"Nonfinite embeddings in {len(dropped)} common wells; clean explicitly")
    if dropped:
        ordered = [key for key, good in zip(ordered, finite) if good]
        aligned = {name: values[finite] for name, values in aligned.items()}
    if not ordered:
        raise ValueError("No finite common wells remain")
    rows = [dict(metadata[key]) for key in ordered]
    return aligned, rows, {
        "n_metadata": len(metadata), "n_requested": len(selected),
        "n_common_before_finite": len(common), "n_common": len(rows),
        "coverage": len(rows) / len(selected), "dropped_nonfinite_well_ids": dropped,
        "models": model_report,
    }


def _dose(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Invalid concentration: {text!r}") from error
    if not number.is_finite():
        raise ValueError(f"Nonfinite concentration: {text!r}")
    return str(number.normalize())


def evaluation_metadata(rows: Sequence[Mapping]) -> list[dict]:
    """Define query labels conservatively from verified RxRx3 control types.

    Only Query guides and Query Compounds + Intron control count as biological
    queries. Controls, positive controls and missing annotations never become
    a shared positive class. A batch is experiment + plate, not plate alone.
    """
    result = []
    seen = set()
    for row in rows:
        item = dict(row)
        key = item.get("well_id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate metadata well_id")
        seen.add(key)
        experiment = str(item.get("experiment_name", "")).strip()
        plate = str(item.get("plate", "")).strip()
        if not experiment or not plate:
            raise ValueError(f"Missing experiment or plate for {key}")
        kind = str(item.get("perturbation_type", "")).upper()
        treatment = str(item.get("treatment", "")).strip()
        gene = str(item.get("gene", "")).strip()
        annotation = str(item.get("well_type_label", "")).strip()
        label = None
        if kind == "CRISPR" and annotation == "Query guides" and gene and treatment:
            label = f"gene:{gene}"
        elif (kind == "COMPOUND" and annotation == "Query Compounds + Intron control"
              and treatment and str(item.get("concentration", "")).strip()):
            label = f"compound:{treatment}|dose:{_dose(item['concentration'])}"
        item.update(batch_id=f"{experiment}/{plate}", perturbation_label=label,
                    is_query=label is not None, dose=_dose(item.get("concentration", "")),
                    guide=treatment if kind == "CRISPR" else "")
        result.append(item)
    return result


def unit_vectors(values: np.ndarray, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    """Return normalized vectors and validity; constant Pearson vectors are invalid.

    Invalid rows become zero internally and MUST be excluded using the returned
    mask. They are never silently interpreted as a valid similarity of zero.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1]:
        raise ValueError("Expected a two-dimensional, nonempty feature matrix")
    if metric not in {"cosine", "pearson"}:
        raise ValueError("metric must be cosine or pearson")
    valid = np.isfinite(matrix).all(axis=1)
    clean = np.where(np.isfinite(matrix), matrix, 0.0)
    if metric == "pearson":
        clean -= clean.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(clean, axis=1)
    valid &= norms > np.finfo(np.float64).eps
    clean /= np.where(valid, norms, 1.0)[:, None]
    clean[~valid] = 0.0
    return clean.astype(np.float32), valid


def _inputs(values: np.ndarray, rows: Sequence[Mapping], metric: str):
    meta = evaluation_metadata(rows)
    if len(values) != len(meta):
        raise ValueError("Embedding and metadata row counts differ; align by well_id first")
    normalized, valid = unit_vectors(values, metric)
    return meta, normalized, valid


def _selection(rows, keys: Iterable[str] | None) -> np.ndarray:
    if keys is None:
        return np.ones(len(rows), dtype=bool)
    wanted = set(keys)
    if not wanted <= {row["well_id"] for row in rows}:
        raise ValueError("Query/candidate selection contains wells absent from aligned rows")
    return np.array([row["well_id"] in wanted for row in rows], dtype=bool)


def _ks(ks):
    ks = tuple(sorted(set(ks)))
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    return ks


def _rank_record(scores, candidate_ids, positives, ks):
    order = np.lexsort((np.asarray(candidate_ids, dtype=str), -np.asarray(scores)))
    relevance = np.asarray(positives, dtype=bool)[order]
    positive_ranks = np.flatnonzero(relevance) + 1
    n_pos, n_total = len(positive_ranks), len(relevance)
    ap = float(np.mean(np.arange(1, n_pos + 1) / positive_ranks))
    harmonic = np.sum(1.0 / np.arange(1, n_total + 1))
    random_ap = float(harmonic / n_total + (n_pos - 1) / (n_total - 1)
                      * (1 - harmonic / n_total)) if n_total > 1 else 1.0
    result = {"ap": ap, "mrr": float(1.0 / positive_ranks[0]),
              "n_candidates": n_total, "n_positives": n_pos, "random_ap": random_ap,
              "first_positive_rank": int(positive_ranks[0])}
    for k in ks:
        n_hits = int(relevance[:k].sum())
        result[f"recall@{k}"] = n_hits / n_pos
        result[f"hit@{k}"] = float(n_hits > 0)
        result[f"random_recall@{k}"] = min(k, n_total) / n_total
    return result


def _summarize(records, n_queries, ks, *, seed, n_bootstrap, extra=None):
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    valid = [record for record in records if record.get("reason") is None]
    fields = ["ap", "mrr", "random_ap", *[f"recall@{k}" for k in ks],
              *[f"hit@{k}" for k in ks], *[f"random_recall@{k}" for k in ks]]
    grouped = defaultdict(list)
    for record in valid:
        grouped[record["perturbation_label"]].append(record)
    groups = [{"perturbation_label": label, "n_query_tasks": len(items),
               **{field: float(np.mean([item[field] for item in items])) for field in fields}}
              for label, items in sorted(grouped.items())]
    summary = {"n_queries": n_queries, "n_evaluable_queries": len({r["well_id"] for r in valid}),
               "n_query_tasks": len(records), "n_evaluable_tasks": len(valid),
               "n_groups": len(groups), "coverage": len({r["well_id"] for r in valid}) / n_queries
               if n_queries else 0.0,
               "excluded_reasons": dict(Counter(r["reason"] for r in records if r.get("reason"))),
               "aggregation": "macro mean across perturbation labels; tasks averaged within label",
               "recall_definition": "fraction of all relevant candidates in the first K ranks",
               "tie_break": "ascending candidate well_id (gene name for gene prototypes)",
               "bootstrap_unit": "perturbation_label", "seed": seed}
    for field in fields:
        summary["map" if field == "ap" else field] = float(np.mean([g[field] for g in groups])) if groups else None
    summary["candidate_count_range"] = [min(r["n_candidates"] for r in valid),
                                         max(r["n_candidates"] for r in valid)] if valid else None
    summary["ci95"] = {}
    if groups and n_bootstrap:
        rng = np.random.default_rng(seed)
        data = np.array([[group[field] for field in fields] for group in groups])
        boot = np.empty((n_bootstrap, len(fields)))
        for i in range(n_bootstrap):
            boot[i] = data[rng.integers(0, len(data), len(data))].mean(axis=0)
        for i, field in enumerate(fields):
            summary["ci95"]["map" if field == "ap" else field] = np.quantile(boot[:, i], [0.025, 0.975]).tolist()
    if extra:
        summary.update(extra)
    return {"summary": summary, "queries": records, "groups": groups}


def retrieval_metrics(
    values: np.ndarray, rows: Sequence[Mapping], *, metric: str = "cosine",
    ks: Sequence[int] = (1, 5, 10), protocol: str = "compound",
    query_well_ids: Iterable[str] | None = None,
    candidate_well_ids: Iterable[str] | None = None, block_size: int = 64,
    seed: int = 2026, n_bootstrap: int = 1000,
) -> dict:
    """Same-experiment/same-dose, cross-plate compound retrieval (Project §6.1).

    ``protocol='gene_well'`` is a supplementary same-experiment cross-plate,
    cross-guide retrieval. Use gene_retrieval_metrics for the primary gene
    prototype protocol. Controls never count as queries or gallery positives.
    Scores are computed separately per experiment/dose context. Peak similarity
    storage is block_size × context gallery size, never N × N.
    """
    if block_size < 1 or protocol not in {"compound", "gene_well"}:
        raise ValueError("Invalid block_size or retrieval protocol")
    ks = _ks(ks)
    meta, vectors, valid = _inputs(values, rows, metric)
    kind = "COMPOUND" if protocol == "compound" else "CRISPR"
    query = _selection(meta, query_well_ids)
    gallery = _selection(meta, candidate_well_ids)
    eligible = np.array([r["is_query"] and r["perturbation_type"].upper() == kind for r in meta])
    q_indices = np.flatnonzero(query & eligible)
    candidate_pool = gallery & eligible & valid
    ids = np.array([r["well_id"] for r in meta], dtype=object)
    batches = np.array([r["batch_id"] for r in meta], dtype=object)
    guides = np.array([r["guide"] for r in meta], dtype=object)
    labels = np.array([r["perturbation_label"] for r in meta], dtype=object)
    query_contexts, gallery_contexts = defaultdict(list), defaultdict(list)
    for i, row in enumerate(meta):
        context = row["experiment_name"], row["dose"] if kind == "COMPOUND" else ""
        if query[i] and eligible[i]:
            query_contexts[context].append(i)
        if candidate_pool[i]:
            gallery_contexts[context].append(i)
    records = []
    for context, context_queries in sorted(query_contexts.items()):
        gallery_indices = np.array(gallery_contexts[context], dtype=int)
        context_vectors = vectors[gallery_indices]
        context_ids, context_batches = ids[gallery_indices], batches[gallery_indices]
        context_guides, context_labels = guides[gallery_indices], labels[gallery_indices]
        for start in range(0, len(context_queries), block_size):
            selected = context_queries[start:start + block_size]
            similarities = vectors[selected] @ context_vectors.T
            for qi, scores in zip(selected, similarities):
                record = {"well_id": ids[qi], "perturbation_label": labels[qi], "reason": None}
                if not valid[qi]:
                    record["reason"] = "invalid_query_vector"
                    records.append(record)
                    continue
                mask = (context_batches != batches[qi]) & (context_ids != ids[qi])
                if kind == "CRISPR":
                    mask &= context_guides != guides[qi]
                positive = context_labels[mask] == labels[qi]
                if not positive.any():
                    record["reason"] = "no_cross_plate_positives"
                elif positive.all():
                    record["reason"] = "no_negative_candidates"
                else:
                    record.update(_rank_record(scores[mask], context_ids[mask], positive, ks))
                records.append(record)
    records.sort(key=lambda record: record["well_id"])
    return _summarize(records, len(q_indices), ks, seed=seed, n_bootstrap=n_bootstrap,
                      extra={"protocol": protocol, "metric": metric, "block_size": block_size,
                             "n_invalid_vectors": int((~valid).sum())})


def gene_retrieval_metrics(
    values: np.ndarray, rows: Sequence[Mapping], *, metric: str = "cosine",
    ks: Sequence[int] = (1, 5), query_well_ids: Iterable[str] | None = None,
    candidate_well_ids: Iterable[str] | None = None, block_size: int = 64,
    seed: int = 2026, n_bootstrap: int = 1000,
) -> dict:
    """Cross-experiment, cross-guide gene prototype retrieval (Project §6.1).

    Each candidate experiment is a separate task with one raw-embedding mean
    prototype per gene. The query's entire guide is removed before computing
    prototypes, not merely its well. Experiments without the query gene cannot
    supply a positive and are omitted from tasks; zero-task queries are recorded.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    ks = _ks(ks)
    meta, vectors, valid = _inputs(values, rows, metric)
    query = _selection(meta, query_well_ids)
    gallery = _selection(meta, candidate_well_ids)
    eligible = np.array([r["is_query"] and r["perturbation_type"].upper() == "CRISPR" for r in meta])
    q_indices = np.flatnonzero(query & eligible)
    candidates = np.flatnonzero(gallery & eligible & valid)
    raw = np.asarray(values, dtype=np.float64)
    by_exp = defaultdict(list)
    for index in candidates:
        by_exp[meta[index]["experiment_name"]].append(index)
    pools = {}
    gene_experiments = defaultdict(set)
    for experiment, indices in by_exp.items():
        genes = sorted({meta[i]["gene"] for i in indices})
        gene_to_i = {gene: i for i, gene in enumerate(genes)}
        sums = np.zeros((len(genes), raw.shape[1]), dtype=np.float64)
        counts = np.zeros(len(genes), dtype=int)
        guide_sums = {}
        for i in indices:
            gi, guide = gene_to_i[meta[i]["gene"]], meta[i]["guide"]
            sums[gi] += raw[i]
            counts[gi] += 1
            if guide not in guide_sums:
                guide_sums[guide] = [gi, np.zeros(raw.shape[1]), 0]
            if guide_sums[guide][0] != gi:
                raise ValueError("A guide maps to multiple genes")
            guide_sums[guide][1] += raw[i]
            guide_sums[guide][2] += 1
        pools[experiment] = genes, sums, counts, guide_sums
        for gene in genes:
            gene_experiments[gene].add(experiment)
    task_groups = defaultdict(list)
    records = []
    for qi in q_indices:
        row = meta[qi]
        record = {"well_id": row["well_id"], "perturbation_label": row["perturbation_label"], "reason": None}
        if not valid[qi]:
            records.append({**record, "reason": "invalid_query_vector"})
            continue
        experiments = gene_experiments[row["gene"]] - {row["experiment_name"]}
        if not experiments:
            records.append({**record, "reason": "no_cross_experiment_positives"})
        for experiment in sorted(experiments):
            task_groups[experiment, row["guide"]].append(qi)
    for (experiment, guide), indices in sorted(task_groups.items()):
        genes, base_sums, base_counts, guide_sums = pools[experiment]
        sums, counts = base_sums.copy(), base_counts.copy()
        if guide in guide_sums:
            gi, total, count = guide_sums[guide]
            sums[gi] -= total
            counts[gi] -= count
        means = sums / np.maximum(counts, 1)[:, None]
        prototypes, good = unit_vectors(means, metric)
        good &= counts > 0
        selected = np.flatnonzero(good)
        candidate_genes = [genes[i] for i in selected]
        for start in range(0, len(indices), block_size):
            block = indices[start:start + block_size]
            scores = vectors[block] @ prototypes[selected].T
            for qi, similarities in zip(block, scores):
                row = meta[qi]
                record = {"well_id": row["well_id"], "perturbation_label": row["perturbation_label"],
                          "candidate_experiment": experiment, "reason": None}
                positive = np.array([gene == row["gene"] for gene in candidate_genes])
                if not positive.any():
                    record["reason"] = "no_cross_guide_positives"
                elif positive.all():
                    record["reason"] = "no_negative_candidates"
                else:
                    record.update(_rank_record(similarities, candidate_genes, positive, ks))
                records.append(record)
    return _summarize(records, len(q_indices), ks, seed=seed, n_bootstrap=n_bootstrap,
                      extra={"protocol": "cross_experiment_cross_guide_gene_prototypes", "metric": metric,
                             "n_invalid_vectors": int((~valid).sum()),
                             "prototype": "mean raw embeddings after excluding the query guide"})


def percent_replicating(
    values: np.ndarray, rows: Sequence[Mapping], *, metric: str = "pearson",
    n_null: int = 500, n_bootstrap: int = 1000, seed: int = 2026,
    max_replicates: int | None = 20, max_groups: int | None = None,
) -> dict:
    """Median cross-plate replicate similarity vs a repeat-count matched null.

    Groups are (perturbation label, experiment). Each null group uses the same
    number of wells and the identical plate/type/dose pattern, drawn from
    mutually distinct OTHER perturbation labels. For CRISPR, observed pairs
    additionally exclude the same guide. A group replicates if its observed
    median is strictly above the null 95th percentile. This precise variant is
    reported explicitly; it is not interchangeable with every published PR
    implementation. Percentages and bootstrap CIs are on a 0..100 scale.
    """
    if n_null < 1 or n_bootstrap < 0:
        raise ValueError("n_null must be positive and n_bootstrap nonnegative")
    if max_replicates is not None and max_replicates < 2:
        raise ValueError("max_replicates must be >= 2 or None")
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive or None")
    meta, vectors, valid = _inputs(values, rows, metric)
    rng = np.random.default_rng(seed)
    groups, pools = defaultdict(list), defaultdict(list)
    for i, row in enumerate(meta):
        if row["is_query"] and valid[i]:
            groups[row["perturbation_label"], row["experiment_name"]].append(i)
            pools[row["batch_id"], row["perturbation_type"], row["dose"]].append(i)
    all_groups = sorted(groups)
    selected_groups = all_groups
    if max_groups is not None and len(all_groups) > max_groups:
        picks = sorted(rng.choice(len(all_groups), max_groups, replace=False).tolist())
        selected_groups = [all_groups[i] for i in picks]
    records = []
    for label, experiment in selected_groups:
        indices = groups[label, experiment]
        original_n = len(indices)
        if max_replicates is not None and original_n > max_replicates:
            indices = sorted(rng.choice(indices, max_replicates, replace=False).tolist())
        pair_i, pair_j = np.triu_indices(len(indices), 1)
        allowed = np.array([meta[indices[a]]["batch_id"] != meta[indices[b]]["batch_id"]
                            and (meta[indices[a]]["perturbation_type"] != "CRISPR"
                                 or meta[indices[a]]["guide"] != meta[indices[b]]["guide"])
                            for a, b in zip(pair_i, pair_j)], dtype=bool)
        pair_i, pair_j = pair_i[allowed], pair_j[allowed]
        record = {"perturbation_label": label, "experiment_name": experiment,
                  "n_replicates_available": original_n, "n_replicates_used": len(indices),
                  "n_pairs": len(pair_i), "reason": None}
        if not len(pair_i):
            records.append({**record, "reason": "no_cross_plate_replicate_pairs"})
            continue
        block = vectors[indices]
        observed = float(np.median(np.einsum("ij,ij->i", block[pair_i], block[pair_j])))
        choices = []
        for i in indices:
            row = meta[i]
            choices.append(np.array([j for j in pools[row["batch_id"], row["perturbation_type"], row["dose"]]
                                     if meta[j]["perturbation_label"] != label], dtype=int))
        null = []
        for _ in range(n_null * 5):
            used = set()
            sample = []
            # Constrained slots first reduces failures for unbalanced plates.
            sample_by_slot = {}
            for slot in sorted(range(len(choices)), key=lambda s: len(choices[s])):
                available = [j for j in choices[slot] if meta[j]["perturbation_label"] not in used]
                if not available:
                    break
                j = int(rng.choice(available))
                used.add(meta[j]["perturbation_label"])
                sample_by_slot[slot] = j
            if len(sample_by_slot) != len(indices):
                continue
            sample = [sample_by_slot[slot] for slot in range(len(indices))]
            block = vectors[sample]
            null.append(float(np.median(np.einsum("ij,ij->i", block[pair_i], block[pair_j]))))
            if len(null) == n_null:
                break
        if len(null) != n_null:
            records.append({**record, "reason": "insufficient_matched_distinct_label_null",
                            "n_null": len(null), "median_similarity": observed})
            continue
        threshold = float(np.quantile(null, 0.95))
        records.append({**record, "median_similarity": observed, "null_q95": threshold,
                        "n_null": n_null, "replicating": bool(observed > threshold),
                        "p_value": (1 + sum(value >= observed for value in null)) / (n_null + 1)})
    by_label = defaultdict(list)
    for record in records:
        if record["reason"] is None:
            by_label[record["perturbation_label"]].append(float(record["replicating"]))
    scores = np.array([np.mean(items) for _, items in sorted(by_label.items())])
    score = float(100 * scores.mean()) if len(scores) else None
    ci = None
    if len(scores) and n_bootstrap:
        boot = [100 * scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(n_bootstrap)]
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"summary": {"percent_replicating": score, "ci95": ci,
                         "n_groups_available": len(all_groups), "n_groups_sampled": len(selected_groups),
                         "n_evaluable_groups": sum(r["reason"] is None for r in records),
                         "n_evaluable_perturbations": len(scores),
                         "coverage": sum(r["reason"] is None for r in records) / len(records) if records else 0.0,
                         "excluded_reasons": dict(Counter(r["reason"] for r in records if r["reason"])),
                         "metric": metric, "null_quantile": 0.95, "n_null": n_null,
                         "max_replicates": max_replicates, "max_groups": max_groups, "seed": seed,
                         "bootstrap_unit": "perturbation_label", "n_invalid_vectors": int((~valid).sum()),
                         "null_matching": "repeat count + exact plate/type/dose slots; distinct other labels",
                         "aggregation": "macro perturbation mean, experiments averaged within perturbation"},
            "groups": records}
