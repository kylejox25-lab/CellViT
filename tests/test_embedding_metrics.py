from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cellvit.embedding_metrics import (
    EmbeddingTable, align_embedding_tables, evaluation_metadata,
    gene_retrieval_metrics, load_embedding_parquet, percent_replicating,
    retrieval_metrics, unit_vectors,
)


def compound(well_id, treatment, plate, *, experiment="compound-001", dose="2.5", query=True):
    return {
        "well_id": well_id, "experiment_name": experiment, "plate": str(plate),
        "gene": "", "treatment": treatment, "concentration": dose,
        "perturbation_type": "COMPOUND",
        "well_type_label": "Query Compounds + Intron control" if query else "",
    }


def gene(well_id, target, guide, experiment, plate=1, *, query=True):
    return {
        "well_id": well_id, "experiment_name": experiment, "plate": str(plate),
        "gene": target, "treatment": f"{target}_guide_{guide}", "concentration": "",
        "perturbation_type": "CRISPR",
        "well_type_label": "Query guides" if query else "CRISPR Gene Positive Controls",
    }


def repeated_compounds(n_labels=6, n_plates=3):
    rows, vectors = [], []
    for label in range(n_labels):
        for plate in range(n_plates):
            rows.append(compound(f"c{label}_p{plate}", f"drug{label}", plate))
            vectors.append(np.eye(n_labels)[label])
    return rows, np.asarray(vectors)


def test_ideal_replicates_outperform_random_and_block_size_is_invariant():
    rows, values = repeated_compounds()
    good = retrieval_metrics(values, rows, ks=(1, 5), block_size=1, n_bootstrap=50)
    other = retrieval_metrics(values, rows, ks=(1, 5), block_size=64, n_bootstrap=50)
    random = retrieval_metrics(np.random.default_rng(42).normal(size=values.shape), rows,
                               ks=(1, 5), n_bootstrap=0)
    assert good["summary"]["map"] == 1.0
    assert good["summary"]["recall@1"] == 0.5  # two relevant wells: recall is not hit rate
    assert good["summary"]["hit@1"] == 1.0
    assert good["summary"] == {**other["summary"], "block_size": 1}
    assert good["summary"]["map"] > random["summary"]["map"]
    assert good["summary"]["ci95"]["map"] == [1.0, 1.0]


def test_plate_exclusion_same_dose_and_no_positive_coverage():
    rows = [compound("q", "A", 1), compound("positive", "A", 2, dose="2.50"),
            compound("same_plate", "B", 1), compound("negative", "B", 2),
            compound("wrong_dose", "A", 2, dose="10"),
            compound("wrong_experiment", "A", 2, experiment="compound-002")]
    values = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]])
    result = retrieval_metrics(values, rows, query_well_ids=["q"], n_bootstrap=0)
    assert result["summary"]["map"] == 1
    assert result["queries"][0]["n_candidates"] == 2
    no_positive = retrieval_metrics(values, rows, query_well_ids=["q"],
                                    candidate_well_ids=["same_plate", "negative"], n_bootstrap=0)
    assert no_positive["summary"]["coverage"] == 0
    assert no_positive["summary"]["map"] is None
    assert no_positive["queries"][0]["reason"] == "no_cross_plate_positives"
    with pytest.raises(ValueError, match="absent"):
        retrieval_metrics(values, rows, query_well_ids=["unknown"])


def test_controls_are_not_positive_classes_and_batch_includes_experiment():
    rows = [compound("control1", "EMPTY_control", 1, query=False),
            compound("control2", "EMPTY_control", 2, query=False),
            gene("positive_control", "G", 1, "gene-001", query=False),
            compound("query", "A", 1, experiment="compound-002")]
    meta = evaluation_metadata(rows)
    assert [row["perturbation_label"] for row in meta[:3]] == [None, None, None]
    assert meta[0]["batch_id"] != meta[3]["batch_id"]
    result = retrieval_metrics(np.eye(4), rows, n_bootstrap=0)
    assert result["summary"]["n_queries"] == 1


def test_alignment_is_by_id_reports_intersection_and_rejects_bad_metadata():
    rows = [compound("a", "A", 1), compound("b", "B", 2), compound("c", "C", 3)]
    metadata = {row["well_id"]: row for row in rows}
    tables = {"one": EmbeddingTable(("b", "a", "c"), np.array([[2], [1], [3]])),
              "two": EmbeddingTable(("c", "a"), np.array([[30, 31], [10, 11]]))}
    aligned, joined, report = align_embedding_tables(tables, metadata)
    assert [row["well_id"] for row in joined] == ["a", "c"]
    np.testing.assert_array_equal(aligned["one"], [[1], [3]])
    np.testing.assert_array_equal(aligned["two"], [[10, 11], [30, 31]])
    assert report["models"]["two"]["n_selected_missing"] == 1
    assert report["coverage"] == 2 / 3
    with pytest.raises(ValueError, match="disagree"):
        align_embedding_tables(tables, {"a": rows[1]})
    with pytest.raises(ValueError, match="absent from metadata"):
        align_embedding_tables({"bad": EmbeddingTable(("unknown",), np.ones((1, 2)))}, metadata)
    with pytest.raises(ValueError, match="Duplicate"):
        EmbeddingTable(("a", "a"), np.ones((2, 2)))


def test_nonfinite_removal_is_explicit_and_common_to_all_models():
    metadata = {key: compound(key, key, i) for i, key in enumerate(("a", "b", "c"))}
    tables = {"one": EmbeddingTable(("a", "b", "c"), np.array([[1], [np.nan], [3]])),
              "two": EmbeddingTable(("c", "b", "a"), np.array([[30], [20], [10]]))}
    with pytest.raises(ValueError, match="Nonfinite"):
        align_embedding_tables(tables, metadata)
    aligned, rows, report = align_embedding_tables(tables, metadata, drop_nonfinite=True)
    assert [row["well_id"] for row in rows] == ["a", "c"]
    assert report["dropped_nonfinite_well_ids"] == ["b"]
    np.testing.assert_array_equal(aligned["two"], [[10], [30]])


def test_load_existing_wide_schema_and_native_list_schema(tmp_path: Path):
    wide = tmp_path / "wide.parquet"
    pq.write_table(pa.table({"well_id": ["b", "a"], "feature_10": [10., 11.],
                            "feature_2": [2., 3.], "__index_level_0__": [0, 1]}), wide)
    loaded = load_embedding_parquet(wide, well_ids=["a"], batch_size=1)
    assert loaded.well_ids == ("a",)
    assert loaded.feature_names == ("feature_2", "feature_10")
    np.testing.assert_array_equal(loaded.values, [[3, 11]])
    cellprofiler = tmp_path / "cp.parquet"
    pq.write_table(pa.table({"well_id": ["a"], "AreaShape_Area": [4.],
                            "Texture_1": [5.], "__index_level_0__": [44]}), cellprofiler)
    assert load_embedding_parquet(cellprofiler).values.shape == (1, 2)
    native = tmp_path / "native.parquet"
    pq.write_table(pa.table({"well_id": ["a", "b"], "embedding": pa.array([[1., 2.], [3., 4.]],
                             type=pa.list_(pa.float32(), 2))}), native)
    np.testing.assert_array_equal(load_embedding_parquet(native, batch_size=1).values, [[1, 2], [3, 4]])
    duplicate = tmp_path / "duplicate.parquet"
    pq.write_table(pa.table({"well_id": ["a", "b", "b"], "feature_0": [1., 2., 3.]}), duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_embedding_parquet(duplicate, well_ids=["a"], batch_size=1)


def test_constant_pearson_and_zero_cosine_vectors_have_explicit_coverage():
    values = np.array([[1., 1.], [0., 0.], [1., 2.], [np.nan, 3.]])
    normalized, good = unit_vectors(values, "pearson")
    assert good.tolist() == [False, False, True, False]
    assert np.isfinite(normalized).all()
    assert unit_vectors(values, "cosine")[1].tolist() == [True, False, True, False]
    rows, vectors = repeated_compounds()
    vectors[0] = 0
    result = retrieval_metrics(vectors, rows, n_bootstrap=0)
    assert result["summary"]["excluded_reasons"]["invalid_query_vector"] == 1
    assert result["summary"]["coverage"] == 17 / 18


def test_ties_are_resolved_by_id_and_not_input_order():
    rows = [compound("q", "A", 1), compound("z-positive", "A", 2), compound("a-negative", "B", 2)]
    values = np.ones((3, 2))
    result = retrieval_metrics(values, rows, query_well_ids=["q"], n_bootstrap=0)
    reversed_result = retrieval_metrics(values[::-1], rows[::-1], query_well_ids=["q"], n_bootstrap=0)
    assert result["summary"]["map"] == 0.5
    assert result["queries"] == reversed_result["queries"]


def test_gene_prototypes_remove_entire_query_guide_and_other_experiment_isolated():
    rows = [gene("q", "G", 1, "gene-001"), gene("leak", "G", 1, "gene-002"),
            gene("positive", "G", 2, "gene-002"), gene("negative", "H", 1, "gene-002"),
            gene("other_experiment", "H", 1, "gene-003"),
            gene("control", "G", 3, "gene-002", query=False)]
    values = np.array([[1, 0], [10, 0], [0, 1], [.8, .6], [1, 0], [100, 0]])
    result = gene_retrieval_metrics(values, rows, query_well_ids=["q"], n_bootstrap=0)
    assert result["summary"]["mrr"] == 0.5
    assert result["summary"]["n_query_tasks"] == 1
    assert result["queries"][0]["candidate_experiment"] == "gene-002"
    assert result["queries"][0]["n_candidates"] == 2
    no_cross_guide = gene_retrieval_metrics(values, rows, query_well_ids=["q"],
                                           candidate_well_ids=["leak", "negative"], n_bootstrap=0)
    assert no_cross_guide["summary"]["coverage"] == 0
    assert no_cross_guide["queries"][0]["reason"] == "no_cross_guide_positives"


def test_percent_replicating_repeat_count_matched_null_and_bootstrap():
    rows, values = repeated_compounds(n_labels=6, n_plates=2)
    good = percent_replicating(values, rows, n_null=20, n_bootstrap=30, max_replicates=2)
    assert good["summary"]["percent_replicating"] == 100.0
    assert good["summary"]["ci95"] == [100., 100.]
    assert good["summary"]["n_evaluable_groups"] == 6
    assert all(row["n_replicates_used"] == 2 and row["n_null"] == 20 for row in good["groups"])
    random = percent_replicating(np.random.default_rng(9).normal(size=values.shape), rows,
                                 n_null=30, n_bootstrap=0)
    assert random["summary"]["percent_replicating"] < 100
    capped = percent_replicating(values, rows, n_null=5, n_bootstrap=0, max_groups=3)
    assert capped["summary"]["n_groups_available"] == 6
    assert capped["summary"]["n_groups_sampled"] == 3


def test_percent_replicating_reports_insufficient_null_instead_of_inventing_values():
    rows, values = repeated_compounds(n_labels=2, n_plates=2)
    result = percent_replicating(values, rows, n_null=10, n_bootstrap=0)
    assert result["summary"]["percent_replicating"] is None
    assert result["summary"]["coverage"] == 0
    assert result["summary"]["excluded_reasons"] == {"insufficient_matched_distinct_label_null": 2}
