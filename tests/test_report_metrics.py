"""
tests/test_report_metrics.py
============================
Strict automated verification that every metric reported in the project report
is dynamically read from and strictly matches results/metrics/*.csv.
Fails if any metric in the report diverges from the underlying CSVs.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from scripts.generate_guideline_report import (
    load_all_metrics,
    get_metrics_context,
    get_calce_benchmark_rows,
    get_nasa_benchmark_rows,
    get_ablation_rows,
    get_per_cell_rows,
    get_paired_difference_rows,
    get_deployment_rows,
    get_xai_rows,
)


@pytest.fixture(scope="module")
def metrics():
    return load_all_metrics()


def test_csv_files_exist():
    required_csvs = [
        "results/metrics/full_benchmark_summary.csv",
        "results/metrics/ablation_table.csv",
        "results/metrics/ablation_breakdown.csv",
        "results/metrics/per_cell.csv",
        "results/metrics/paired_cell_differences.csv",
        "results/metrics/xai_feature_relevance.csv",
        "results/metrics/xai_rank_correlations.csv",
        "results/metrics/deployment_metrics.csv",
    ]
    for p in required_csvs:
        assert os.path.exists(p), f"Missing required metrics CSV: {p}"


def test_calce_metrics_match_csv(metrics):
    df_b = metrics["bench"]
    rows = get_calce_benchmark_rows(metrics)
    assert len(rows) > 0, "CALCE benchmark rows must not be empty"

    # Find WD-GPR-SR (Proposed)
    prop_row = next(r for r in rows if "WD-GPR-SR (Proposed)" in r[0])
    csv_match = df_b[(df_b["dataset"] == "CALCE") & (df_b["model"] == "WD_GPR_SR_Proposed")].iloc[0]

    # Verify RMSE formatted string
    expected_rmse = f"{csv_match['RMSE_mean']:.2f}±{csv_match['RMSE_std_cells']:.2f}"
    assert prop_row[2] == expected_rmse, f"CALCE RMSE in report ({prop_row[2]}) != CSV ({expected_rmse})"

    # Verify MAE
    expected_mae = f"{csv_match['MAE_mean']:.2f}±{csv_match['MAE_std_cells']:.2f}"
    assert prop_row[1] == expected_mae, f"CALCE MAE in report ({prop_row[1]}) != CSV ({expected_mae})"

    # Verify PICP recal
    expected_picp = f"{csv_match['PICP_recal_mean']:.1f}%"
    assert prop_row[6] == expected_picp, f"CALCE PICP in report ({prop_row[6]}) != CSV ({expected_picp})"

    # Verify MPIW recal
    expected_mpiw = f"{csv_match['MPIW_recal_mean']:.2f}%"
    assert prop_row[7] == expected_mpiw, f"CALCE MPIW in report ({prop_row[7]}) != CSV ({expected_mpiw})"


def test_nasa_metrics_match_csv(metrics):
    df_b = metrics["bench"]
    rows = get_nasa_benchmark_rows(metrics)
    assert len(rows) > 0, "NASA benchmark rows must not be empty"

    prop_row = next(r for r in rows if "WD-GPR-SR (Proposed)" in r[0])
    csv_match = df_b[(df_b["dataset"] == "NASA") & (df_b["model"] == "WD_GPR_SR_Proposed")].iloc[0]

    expected_rmse = f"{csv_match['RMSE_mean']:.2f}±{csv_match['RMSE_std_cells']:.2f}"
    assert prop_row[2] == expected_rmse, f"NASA RMSE in report ({prop_row[2]}) != CSV ({expected_rmse})"

    expected_mae = f"{csv_match['MAE_mean']:.2f}±{csv_match['MAE_std_cells']:.2f}"
    assert prop_row[1] == expected_mae, f"NASA MAE in report ({prop_row[1]}) != CSV ({expected_mae})"

    expected_picp = f"{csv_match['PICP_recal_mean']:.1f}%"
    assert prop_row[6] == expected_picp, f"NASA PICP in report ({prop_row[6]}) != CSV ({expected_picp})"


def test_ablation_metrics_match_csv(metrics):
    df_a = metrics["ablation"]
    rows = get_ablation_rows(metrics)
    assert len(rows) == 5, f"Expected 5 ablation rows (A to E), got {len(rows)}"

    for row in rows:
        label = row[0]
        # Match model key
        if "Sparse GPR Wide" in label:
            m_key = "Sparse_GPR_Wide"
        elif "Deep Kernel GPR" in label:
            m_key = "Deep_Kernel_GPR"
        elif "WD-GPR (no SR)" in label:
            m_key = "WD_GPR"
        elif "Naive SR" in label:
            m_key = "WD_GPR_Unweighted_SR"
        elif "WD-GPR-SR (Prop.)" in label:
            m_key = "WD_GPR_SR_Proposed"
        else:
            pytest.fail(f"Unknown ablation row: {label}")

        c_match = df_a[(df_a["dataset"] == "CALCE") & (df_a["model"] == m_key)].iloc[0]
        n_match = df_a[(df_a["dataset"] == "NASA") & (df_a["model"] == m_key)].iloc[0]

        expected_c_rmse = f"{c_match['RMSE_mean']:.2f}±{c_match['RMSE_std_cells']:.2f}"
        expected_n_rmse = f"{n_match['RMSE_mean']:.2f}±{n_match['RMSE_std_cells']:.2f}"
        assert row[2] == expected_c_rmse, f"Ablation {label} CALCE RMSE mismatch: {row[2]} vs {expected_c_rmse}"
        assert row[4] == expected_n_rmse, f"Ablation {label} NASA RMSE mismatch: {row[4]} vs {expected_n_rmse}"


def test_per_cell_metrics_match_csv(metrics):
    df_pc = metrics["per_cell"]
    rows = get_per_cell_rows(metrics)
    assert len(rows) == 8, f"Expected 8 cells (4 CALCE + 4 NASA), got {len(rows)}"

    for row in rows:
        cid, ds, mae, rmse, r2, picp, mpiw, _ = row
        match = df_pc[(df_pc["cell"] == cid) & (df_pc["dataset"] == ds) & (df_pc["model"] == "WD_GPR_SR_Proposed")].iloc[0]

        assert rmse == f"{match['RMSE']:.2f}", f"Per-cell RMSE mismatch for {cid}: {rmse} vs {match['RMSE']:.2f}"
        assert mae == f"{match['MAE']:.2f}", f"Per-cell MAE mismatch for {cid}: {mae} vs {match['MAE']:.2f}"
        assert r2 == f"{match['R2']:.3f}", f"Per-cell R2 mismatch for {cid}: {r2} vs {match['R2']:.3f}"
        assert picp == f"{match['PICP_recal']:.1f}%", f"Per-cell PICP mismatch for {cid}: {picp} vs {match['PICP_recal']:.1f}%"


def test_paired_differences_match_csv(metrics):
    df_p = metrics["paired"]
    rows = get_paired_difference_rows(metrics)
    assert len(rows) == len(df_p), f"Row count mismatch in paired differences: {len(rows)} vs {len(df_p)}"

    for row in rows:
        cell, ds, e, d_c, d_d, bl_str, _ = row
        match = df_p[(df_p["cell"] == cell) & (df_p["dataset"] == ds)].iloc[0]

        assert e == f"{match['rmse_E']:.2f}"
        assert d_c == f"{match['diff_E_minus_C']:+.2f}"
        assert d_d == f"{match['diff_E_minus_D']:+.2f}"


def test_deployment_metrics_match_csv(metrics):
    df_d = metrics["deployment"]
    rows = get_deployment_rows(metrics)
    assert len(rows) == len(df_d), f"Row count mismatch in deployment: {len(rows)} vs {len(df_d)}"

    for row in rows:
        met_name, met_val, notes = row
        match = df_d[df_d["metric"] == met_name].iloc[0]
        expected_val = f"{match['value']} {match['unit']}"
        assert met_val == expected_val, f"Deployment {met_name} mismatch: {met_val} vs {expected_val}"


def test_no_forbidden_phrases_in_pdf():
    pdf_path = "2401AI38_MLProject.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not built yet")
    import pymupdf
    doc = pymupdf.open(pdf_path)
    full_text = " ".join([p.get_text() for p in doc])

    assert "zero data leakage" not in full_text.lower(), "Found forbidden phrase 'zero data leakage' in PDF"
    assert "rank 1" not in full_text.lower(), "Found forbidden phrase 'Rank 1' in PDF"



def test_ta_report_12to15_pages():
    pdf_path = "2401AI38_MLProject_Report.pdf"
    assert os.path.exists(pdf_path), f"{pdf_path} must exist"
    import pymupdf
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    assert 12 <= page_count <= 15, f"TA Report page count must be between 12 and 15 pages, got {page_count}"
    assert page_count == 14, f"Expected exactly 14 pages, got {page_count}"
    full_text = " ".join([p.get_text() for p in doc])
    assert "zero data leakage" not in full_text.lower()
    assert "rank 1" not in full_text.lower()
