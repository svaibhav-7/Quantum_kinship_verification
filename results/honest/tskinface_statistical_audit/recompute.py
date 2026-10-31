"""Independently audit the corrected TSKinFace result artifacts.

Run from any directory with the project's .venv:
    python results/honest/tskinface_statistical_audit/recompute.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "honest"
OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"
CORPORA = ("FIW", "KinFaceW-I", "KinFaceW-II", "TSKinFace")
BACKBONES = ("FaceNet", "ArcFace")
WIDTH_ARMS = ("ctl-width-linear", "ctl-width-mlp", "ctl-width-random")
AMPLITUDE_ARMS = (
    "amp-expectation",
    "amp-fidelity",
    "control",
    *WIDTH_ARMS,
)


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save_json(name, data):
    (OUT / name).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(name, rows):
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(filename, columns):
    with (TABLES / filename).open(
        "r", newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))
    headers = [label for label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [row[key].replace("|", r"\|").replace("\n", " ") for _, key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paired(x, y):
    """Paired t test, t-based 95% CI for the mean difference x - y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError(f"Expected equally sized paired vectors, got {x.shape}, {y.shape}")
    differences = x - y
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / np.sqrt(n)
    t_value = float(mean / se) if se else 0.0
    p_value = float(2 * stats.t.sf(abs(t_value), n - 1)) if se else 1.0
    margin = float(stats.t.ppf(0.975, n - 1) * se)
    return {
        "n": int(n),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95": [mean - margin, mean + margin],
        "t": t_value,
        "p": p_value,
        "differences": differences.tolist(),
    }


def holm_bh(p_values):
    """Return familywise Holm and Benjamini-Hochberg adjusted p values."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    count = len(values)
    holm = np.empty(count)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (count - rank) * values[index]))
        holm[index] = previous
    bh = np.empty(count)
    previous = 1.0
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        previous = min(previous, values[index] * count / (rank + 1))
        bh[index] = previous
    return holm.tolist(), bh.tolist()


def roc_rows(artifact, corpus, arm):
    return np.asarray(artifact[corpus][arm]["per_fold"], dtype=float)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    corrected_kfold = load("kfold_tskinface_corrected.json")
    historical_kfold = load("kfold.json")
    corrected_vqc = load("vqc_tskinface_corrected.json")["TSKinFace"]
    corrected_amp = {
        "FaceNet": load("amplitude_vqc_tskinface_facenet_corrected.json")[
            "TSKinFace"
        ],
        "ArcFace": load("amplitude_vqc_tskinface_arcface_corrected.json")[
            "TSKinFace"
        ],
    }
    corrected_setlevel = load("setlevel_tskinface_corrected.json")["TSKinFace"]
    old_setlevel = load("setlevel.json")["TSKinFace"]
    old_backbone = load("backbone_comparison.json")
    new_backbone = load("backbone_comparison_tskinface_corrected.json")
    old_readout_summary = load("readout_summary.json")
    corrected_composite_artifact = load("readout_summary_tskinface_corrected.json")

    corrected_artifact_names = [
        "kfold_tskinface_corrected.json",
        "vqc_tskinface_corrected.json",
        "amplitude_vqc_tskinface_facenet_corrected.json",
        "amplitude_vqc_tskinface_arcface_corrected.json",
        "setlevel_tskinface_corrected.json",
        "backbone_comparison_tskinface_corrected.json",
    ]
    corrected_metadata_checks = []
    for filename in corrected_artifact_names:
        artifact = load(filename)
        config = artifact.get("_config", {})
        protocol = config.get("tskinface_protocol", {})
        if config.get("source_commit") != "a480275269855e3ecffe7b3f922793e9b57d97e5":
            raise ValueError(f"Unexpected source commit in {filename}")
        if (
            protocol.get("positive_pairs_after_embedding_coverage") != 1118
            or protocol.get("families_after_embedding_coverage") != 559
            or protocol.get("family_cap_applied") is not False
        ):
            raise ValueError(f"TSKinFace coverage/protocol metadata mismatch in {filename}")
        corrected_metadata_checks.append(
            {
                "artifact": filename,
                "source_commit": config["source_commit"],
                "positive_pairs_after_embedding_coverage": protocol[
                    "positive_pairs_after_embedding_coverage"
                ],
                "families_after_embedding_coverage": protocol[
                    "families_after_embedding_coverage"
                ],
                "family_cap_applied": protocol["family_cap_applied"],
            }
        )

    old_amp_sources = {
        "FaceNet": {
            "FIW": "amp_widthctl_all.json",
            "KinFaceW-I": "amp_widthctl.json",
            "KinFaceW-II": "amp_widthctl_all.json",
            "TSKinFace": "amp_widthctl_all.json",
        },
        "ArcFace": {corpus: "amp_arcface.json" for corpus in CORPORA},
    }
    amp_by_backbone = {}
    amp_source_names = {}
    for backbone in BACKBONES:
        amp_by_backbone[backbone] = {}
        amp_source_names[backbone] = {}
        for corpus in CORPORA:
            if corpus == "TSKinFace":
                amp_by_backbone[backbone][corpus] = corrected_amp[backbone]
                amp_source_names[backbone][corpus] = (
                    f"amplitude_vqc_tskinface_{backbone.lower()}_corrected.json"
                )
            else:
                filename = old_amp_sources[backbone][corpus]
                amp_by_backbone[backbone][corpus] = load(filename)[corpus]
                amp_source_names[backbone][corpus] = filename

    fold_metric_checks = []
    for backbone in BACKBONES:
        rows = amp_by_backbone[backbone]["TSKinFace"]
        for arm in AMPLITUDE_ARMS:
            values = np.asarray(rows[arm]["per_fold"], dtype=float)
            mean, sd = mean_sd(values)
            if not np.isclose(mean, rows[arm]["roc_auc_mean"], atol=1e-12):
                raise ValueError(f"Amplitude AUC mean mismatch: {backbone} {arm}")
            if not np.isclose(sd, rows[arm]["roc_auc_sd"], atol=1e-12):
                raise ValueError(f"Amplitude AUC SD mismatch: {backbone} {arm}")
            fold_metric_checks.append(
                {
                    "artifact": amp_source_names[backbone]["TSKinFace"],
                    "model": arm,
                    "folds": len(values),
                    "recomputed_AUC_mean": mean,
                    "stored_AUC_mean": rows[arm]["roc_auc_mean"],
                    "recomputed_AUC_SD": sd,
                    "stored_AUC_SD": rows[arm]["roc_auc_sd"],
                }
            )
    for arm, row in corrected_vqc.items():
        values = np.asarray(row["per_fold"], dtype=float)
        mean, sd = mean_sd(values)
        if not np.isclose(mean, row["roc_auc_mean"], atol=1e-12):
            raise ValueError(f"Standard VQC AUC mean mismatch: {arm}")
        if not np.isclose(sd, row["roc_auc_sd"], atol=1e-12):
            raise ValueError(f"Standard VQC AUC SD mismatch: {arm}")
        fold_metric_checks.append(
            {
                "artifact": "vqc_tskinface_corrected.json",
                "model": arm,
                "folds": len(values),
                "recomputed_AUC_mean": mean,
                "stored_AUC_mean": row["roc_auc_mean"],
                "recomputed_AUC_SD": sd,
                "stored_AUC_SD": row["roc_auc_sd"],
            }
        )

    # Validate that the hand-selected historical artifacts reproduce the
    # previously stored scalar/vector summary before substituting corrected data.
    for backbone in BACKBONES:
        for corpus in CORPORA:
            historical_source = load(old_amp_sources[backbone][corpus])[corpus]
            old_summary = old_readout_summary["per_cell"][backbone][corpus]
            vector = historical_source["amp-expectation"]["roc_auc_mean"]
            scalar = historical_source["amp-fidelity"]["roc_auc_mean"]
            if not np.isclose(vector, old_summary["vector"], atol=1e-12):
                raise ValueError(f"Historical vector mismatch: {backbone} {corpus}")
            if not np.isclose(scalar, old_summary["scalar"], atol=1e-12):
                raise ValueError(f"Historical scalar mismatch: {backbone} {corpus}")

    provenance = {
        "corrected_source_commit": "a480275269855e3ecffe7b3f922793e9b57d97e5",
        "corrected_artifacts": [
            "kfold_tskinface_corrected.json",
            "vqc_tskinface_corrected.json",
            "amplitude_vqc_tskinface_facenet_corrected.json",
            "amplitude_vqc_tskinface_arcface_corrected.json",
            "setlevel_tskinface_corrected.json",
            "backbone_comparison_tskinface_corrected.json",
        ],
        "historical_artifacts": sorted(
            {
                "kfold.json",
                "setlevel.json",
                "backbone_comparison.json",
                "readout_summary.json",
                *old_amp_sources["FaceNet"].values(),
                *old_amp_sources["ArcFace"].values(),
            }
        ),
        "test_method": "Two-sided paired Student t tests; 95% t intervals on paired fold differences.",
        "fold_inference_caveat": (
            "Five folds partition the same corpus and are not independent population draws. "
            "Fold-level p-values and intervals are descriptive/optimistic; corpus-level "
            "summaries are included as the more conservative unit."
        ),
        "historical_alignment_caveat": (
            "Historical JSONs do not store source commits or fold manifests. "
            "Historical-versus-corrected fold-index pairing assumes the shared seed/order "
            "and is exploratory, not independently verified."
        ),
        "historical_composite_caveat": (
            "The corrected composite replaces only TSKinFace folds. Other corpus folds "
            "remain from historical artifacts and were not rerun."
        ),
    }

    # Scalar/vector, all eight backbone x corpus cells.
    readout_rows = []
    readout_tests = []
    per_corpus_deltas = {}
    readout_fold_vectors = {}
    best_control_fold_tests = []
    all_control_tests = []
    for backbone in BACKBONES:
        per_corpus_deltas[backbone] = {}
        readout_fold_vectors[backbone] = {}
        for corpus in CORPORA:
            rows = amp_by_backbone[backbone][corpus]
            vector = roc_rows({corpus: rows}, corpus, "amp-expectation")
            scalar = roc_rows({corpus: rows}, corpus, "amp-fidelity")
            result = paired(vector, scalar)
            readout_tests.append((backbone, corpus, result))
            readout_fold_vectors[backbone][corpus] = {
                "vector": vector.tolist(),
                "scalar": scalar.tolist(),
            }
            per_corpus_deltas[backbone][corpus] = result["mean_difference"]

            old_vector = np.asarray(
                old_readout_summary["per_cell"][backbone][corpus]["vector"]
                if corpus != "TSKinFace"
                else load(old_amp_sources[backbone][corpus])[corpus][
                    "amp-expectation"
                ]["roc_auc_mean"],
                dtype=float,
            )
            old_scalar = np.asarray(
                old_readout_summary["per_cell"][backbone][corpus]["scalar"]
                if corpus != "TSKinFace"
                else load(old_amp_sources[backbone][corpus])[corpus][
                    "amp-fidelity"
                ]["roc_auc_mean"],
                dtype=float,
            )
            historical_delta = float(old_vector - old_scalar)

            control_vectors = {
                arm: roc_rows({corpus: rows}, corpus, arm) for arm in WIDTH_ARMS
            }
            selected_arm = max(
                WIDTH_ARMS,
                key=lambda arm: float(control_vectors[arm].mean()),
            )
            best_per_fold = np.max(
                np.column_stack([control_vectors[arm] for arm in WIDTH_ARMS]),
                axis=1,
            )
            selected_test = paired(vector, best_per_fold)
            best_control_fold_tests.append((backbone, corpus, selected_test))
            direct_tests = {}
            for arm, control in control_vectors.items():
                control_test = paired(vector, control)
                direct_tests[arm] = control_test
                all_control_tests.append((backbone, corpus, arm, control_test))

            readout_rows.append(
                {
                    "backbone": backbone,
                    "corpus": corpus,
                    "TSKinFace_status": (
                        "corrected" if corpus == "TSKinFace" else "historical-only"
                    ),
                    "historical_vector_AUC": fmt(old_vector),
                    "historical_scalar_AUC": fmt(old_scalar),
                    "historical_vector_minus_scalar": fmt(historical_delta),
                    "current_vector_AUC": fmt(vector.mean()),
                    "current_scalar_AUC": fmt(scalar.mean()),
                    "current_vector_minus_scalar": fmt(result["mean_difference"]),
                    "change_in_readout_delta": fmt(
                        result["mean_difference"] - historical_delta
                    ),
                    "paired_CI95_low": fmt(result["ci95"][0]),
                    "paired_CI95_high": fmt(result["ci95"][1]),
                    "paired_t": fmt(result["t"]),
                    "raw_p": fmt(result["p"]),
                    "best_control_arm_by_mean": selected_arm,
                    "vector_minus_best_per_fold_control": fmt(
                        selected_test["mean_difference"]
                    ),
                    "best_control_nominal_p": fmt(selected_test["p"]),
                    "readout_positive": result["mean_difference"] > 0,
                    "readout_significant_unadjusted": result["p"] < 0.05,
                    "source_artifact": amp_source_names[backbone][corpus],
                }
            )

    # Multiplicity: reproduce the paper's 16-test family, then also test every
    # width-matched control separately to expose control selection.
    p16 = [test[2]["p"] for test in readout_tests] + [
        test[2]["p"] for test in best_control_fold_tests
    ]
    holm16, bh16 = holm_bh(p16)
    p32 = [test[2]["p"] for test in readout_tests] + [
        test[3]["p"] for test in all_control_tests
    ]
    holm32, bh32 = holm_bh(p32)

    historical_readout_tests = []
    historical_best_control_tests = []
    historical_all_control_tests = []
    for backbone in BACKBONES:
        for corpus in CORPORA:
            historical_rows = load(old_amp_sources[backbone][corpus])[corpus]
            vector = np.asarray(historical_rows["amp-expectation"]["per_fold"])
            scalar = np.asarray(historical_rows["amp-fidelity"]["per_fold"])
            historical_readout_tests.append(paired(vector, scalar)["p"])
            controls = {
                arm: np.asarray(historical_rows[arm]["per_fold"])
                for arm in WIDTH_ARMS
            }
            best_each_fold = np.max(np.column_stack(list(controls.values())), axis=1)
            historical_best_control_tests.append(
                paired(vector, best_each_fold)["p"]
            )
            for arm in WIDTH_ARMS:
                historical_all_control_tests.append(
                    paired(vector, controls[arm])["p"]
                )
    historical_holm16, historical_bh16 = holm_bh(
        historical_readout_tests + historical_best_control_tests
    )
    historical_holm32, historical_bh32 = holm_bh(
        historical_readout_tests + historical_all_control_tests
    )

    for i, row in enumerate(readout_rows):
        row["historical_readout_raw_p"] = fmt(historical_readout_tests[i])
        row["historical_readout_Holm_16"] = fmt(historical_holm16[i])
        row["historical_readout_BH_16"] = fmt(historical_bh16[i])
        row["readout_Holm_16"] = fmt(holm16[i])
        row["readout_BH_16"] = fmt(bh16[i])
        row["readout_Holm_32_all_controls"] = fmt(holm32[i])
        row["readout_BH_32_all_controls"] = fmt(bh32[i])

    control_adjusted = {}
    for i, (backbone, corpus, arm, result) in enumerate(all_control_tests):
        control_adjusted[(backbone, corpus, arm)] = (holm32[8 + i], bh32[8 + i])

    width_rows = []
    for i, (backbone, corpus, arm, result) in enumerate(all_control_tests):
        controls = amp_by_backbone[backbone][corpus]
        control = roc_rows({corpus: controls}, corpus, arm)
        vector = roc_rows({corpus: controls}, corpus, "amp-expectation")
        holm, bh = control_adjusted[(backbone, corpus, arm)]
        width_rows.append(
            {
                "backbone": backbone,
                "corpus": corpus,
                "TSKinFace_status": (
                    "corrected" if corpus == "TSKinFace" else "historical-only"
                ),
                "quantum_arm": "amp-expectation",
                "classical_control": arm,
                "quantum_mean_AUC": fmt(vector.mean()),
                "classical_mean_AUC": fmt(control.mean()),
                "paired_delta": fmt(result["mean_difference"]),
                "CI95_low": fmt(result["ci95"][0]),
                "CI95_high": fmt(result["ci95"][1]),
                "t": fmt(result["t"]),
                "raw_p": fmt(result["p"]),
                "Holm_32_all_controls": fmt(holm),
                "BH_32_all_controls": fmt(bh),
                "source_artifact": amp_source_names[backbone][corpus],
            }
        )
    write_csv("01_scalar_vs_vector_readout.csv", readout_rows)
    write_csv("02_amplitude_vs_width_controls.csv", width_rows)

    # Full composite readout statistics: 20 fold observations per backbone and
    # four independent corpus-level mean differences.
    composite = {}
    for backbone in BACKBONES:
        vectors = np.concatenate(
            [
                np.asarray(readout_fold_vectors[backbone][corpus]["vector"])
                for corpus in CORPORA
            ]
        )
        scalars = np.concatenate(
            [
                np.asarray(readout_fold_vectors[backbone][corpus]["scalar"])
                for corpus in CORPORA
            ]
        )
        fold_test = paired(vectors, scalars)
        corpus_values = np.asarray(
            [per_corpus_deltas[backbone][corpus] for corpus in CORPORA]
        )
        corpus_test = paired(corpus_values, np.zeros_like(corpus_values))
        composite[backbone] = {
            "fold_level_paired_test": fold_test,
            "corpus_level_one_sample_test": corpus_test,
            "corpus_effects": dict(
                (corpus, per_corpus_deltas[backbone][corpus])
                for corpus in CORPORA
            ),
            "positive_corpus_count": int(np.count_nonzero(corpus_values > 0)),
            "corpus_count": len(CORPORA),
        }
        stored = corrected_composite_artifact["pooled"][backbone]
        for key, actual in (
            ("delta", fold_test["mean_difference"]),
            ("p", fold_test["p"]),
        ):
            if not np.isclose(actual, stored[key], rtol=1e-9, atol=1e-12):
                raise ValueError(f"Corrected composite {key} mismatch for {backbone}")
        if not np.allclose(fold_test["ci95"], stored["ci95"], rtol=1e-9, atol=1e-12):
            raise ValueError(f"Corrected composite CI mismatch for {backbone}")
        for corpus in CORPORA:
            stored_cell = corrected_composite_artifact["per_cell"][backbone][corpus]
            current_cell = amp_by_backbone[backbone][corpus]
            for arm, field in (
                ("amp-expectation", "vector"),
                ("amp-fidelity", "scalar"),
            ):
                if not np.isclose(
                    current_cell[arm]["roc_auc_mean"],
                    stored_cell[field],
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Corrected composite per-cell mismatch: {backbone} {corpus} {arm}"
                    )
            calculated_readout = paired(
                roc_rows({corpus: current_cell}, corpus, "amp-expectation"),
                roc_rows({corpus: current_cell}, corpus, "amp-fidelity"),
            )
            for key, actual in (
                ("readout_delta", calculated_readout["mean_difference"]),
                ("readout_p", calculated_readout["p"]),
            ):
                if not np.isclose(
                    actual, stored_cell[key], rtol=1e-9, atol=1e-12
                ):
                    raise ValueError(
                        f"Corrected composite {key} mismatch: {backbone} {corpus}"
                    )

    # Verify the historical composite statistics against its stored JSON.
    historical_pooled_validation = {}
    for backbone in BACKBONES:
        historical_rows = []
        for corpus in CORPORA:
            historical_artifact = load(old_amp_sources[backbone][corpus])[corpus]
            vector = roc_rows({corpus: historical_artifact}, corpus, "amp-expectation")
            scalar = roc_rows({corpus: historical_artifact}, corpus, "amp-fidelity")
            historical_rows.append((vector, scalar))
        v = np.concatenate([r[0] for r in historical_rows])
        s = np.concatenate([r[1] for r in historical_rows])
        calculated = paired(v, s)
        recorded = old_readout_summary["pooled"][backbone]
        if not np.isclose(calculated["mean_difference"], recorded["delta"], atol=1e-12):
            raise ValueError(f"Historical pooled delta mismatch for {backbone}")
        if not np.isclose(calculated["p"], recorded["p"], rtol=1e-9):
            raise ValueError(f"Historical pooled p-value mismatch for {backbone}")
        historical_pooled_validation[backbone] = {
            "recomputed_delta": calculated["mean_difference"],
            "stored_delta": recorded["delta"],
            "recomputed_p": calculated["p"],
            "stored_p": recorded["p"],
            "recomputed_ci95": calculated["ci95"],
            "stored_ci95": recorded["ci95"],
        }

    # Historical versus corrected TSKinFace performance, paired by fold index.
    tskinface_comparison = []
    old_ts_kfold = historical_kfold["TSKinFace"]["per_fold"]
    new_ts_kfold = corrected_kfold["TSKinFace"]["per_fold"]
    for metric in ("roc_auc", "accuracy"):
        old_values = [row[metric] for row in old_ts_kfold]
        new_values = [row[metric] for row in new_ts_kfold]
        result = paired(new_values, old_values)
        tskinface_comparison.append(
            {
                "evaluation": "Grouped k-fold baseline",
                "backbone": "FaceNet",
                "measure": metric,
                "historical_mean": fmt(np.mean(old_values)),
                "corrected_mean": fmt(np.mean(new_values)),
                "corrected_minus_historical": fmt(result["mean_difference"]),
                "paired_CI95_low": fmt(result["ci95"][0]),
                "paired_CI95_high": fmt(result["ci95"][1]),
                "paired_t": fmt(result["t"]),
                "nominal_p": fmt(result["p"]),
                "pairing_status": "Fold-index pairing; historical fold manifest unavailable.",
                "historical_source": "kfold.json",
                "corrected_source": "kfold_tskinface_corrected.json",
            }
        )
    for backbone in BACKBONES:
        old = load(old_amp_sources[backbone]["TSKinFace"])["TSKinFace"]
        new = corrected_amp[backbone]
        for arm in AMPLITUDE_ARMS:
            old_values = roc_rows({"TSKinFace": old}, "TSKinFace", arm)
            new_values = roc_rows({"TSKinFace": new}, "TSKinFace", arm)
            result = paired(new_values, old_values)
            tskinface_comparison.append(
                {
                    "evaluation": f"Amplitude VQC: {arm}",
                    "backbone": backbone,
                    "measure": "ROC-AUC",
                    "historical_mean": fmt(old_values.mean()),
                    "corrected_mean": fmt(new_values.mean()),
                    "corrected_minus_historical": fmt(result["mean_difference"]),
                    "paired_CI95_low": fmt(result["ci95"][0]),
                    "paired_CI95_high": fmt(result["ci95"][1]),
                    "paired_t": fmt(result["t"]),
                    "nominal_p": fmt(result["p"]),
                    "pairing_status": "Fold-index pairing; historical fold manifest unavailable.",
                    "historical_source": old_amp_sources[backbone]["TSKinFace"],
                    "corrected_source": amp_source_names[backbone]["TSKinFace"],
                }
            )
    for backbone in BACKBONES:
        for arm in ("single", "set", "set+density"):
            old_mean = old_backbone[backbone][arm]["TSKinFace"]["mean"]
            new_mean = new_backbone[backbone][arm]["TSKinFace"]["mean"]
            tskinface_comparison.append(
                {
                    "evaluation": f"Backbone comparison: {arm}",
                    "backbone": backbone,
                    "measure": "ROC-AUC",
                    "historical_mean": fmt(old_mean),
                    "corrected_mean": fmt(new_mean),
                    "corrected_minus_historical": fmt(new_mean - old_mean),
                    "paired_CI95_low": "NA",
                    "paired_CI95_high": "NA",
                    "paired_t": "NA",
                    "nominal_p": "NA",
                    "pairing_status": "Not calculable: artifacts store means/SD, not fold scores.",
                    "historical_source": "backbone_comparison.json",
                    "corrected_source": "backbone_comparison_tskinface_corrected.json",
                }
            )
    for arm in ("single", "set", "set+density"):
        old_mean = old_setlevel[arm]["roc_auc_mean"]
        new_mean = corrected_setlevel[arm]["roc_auc_mean"]
        tskinface_comparison.append(
            {
                "evaluation": f"Set-level evaluation: {arm}",
                "backbone": "FaceNet",
                "measure": "ROC-AUC",
                "historical_mean": fmt(old_mean),
                "corrected_mean": fmt(new_mean),
                "corrected_minus_historical": fmt(new_mean - old_mean),
                "paired_CI95_low": "NA",
                "paired_CI95_high": "NA",
                "paired_t": "NA",
                "nominal_p": "NA",
                "pairing_status": (
                    "Not calculable: summary-only artifact. Historical mean set size "
                    "was 169.03 despite one TSKinFace image per identity."
                ),
                "historical_source": "setlevel.json",
                "corrected_source": "setlevel_tskinface_corrected.json",
            }
        )
    write_csv("03_historical_vs_corrected_tskinface.csv", tskinface_comparison)

    # Standard VQC under the corrected protocol versus its local and global
    # classical controls. Global-baseline test folds are aligned by seed, but
    # that model is trained on pooled corpora rather than TSKinFace alone.
    vqc_rows = []
    vqc_vector = np.asarray(corrected_vqc["vqc-expectation"]["per_fold"])
    vqc_scalar = np.asarray(corrected_vqc["vqc-fidelity"]["per_fold"])
    vqc_control = np.asarray(corrected_vqc["control"]["per_fold"])
    baseline_values = np.asarray([row["roc_auc"] for row in new_ts_kfold])
    comparisons = (
        (
            "VQC expectation",
            "capacity-matched VQC control",
            vqc_vector,
            vqc_control,
            "Same held-out folds and VQC protocol.",
        ),
        (
            "VQC fidelity",
            "capacity-matched VQC control",
            vqc_scalar,
            vqc_control,
            "Same held-out folds and VQC protocol.",
        ),
        (
            "VQC expectation",
            "VQC fidelity",
            vqc_vector,
            vqc_scalar,
            "Readout contrast on matched held-out folds.",
        ),
        (
            "VQC expectation",
            "global grouped-k-fold baseline",
            vqc_vector,
            baseline_values,
            "Matched test folds; baseline trained on pooled corpora.",
        ),
        (
            "VQC fidelity",
            "global grouped-k-fold baseline",
            vqc_scalar,
            baseline_values,
            "Matched test folds; baseline trained on pooled corpora.",
        ),
    )
    for arm_a, arm_b, values, control_values, note in comparisons:
        result = paired(values, control_values)
        vqc_rows.append(
            {
                "dataset": "TSKinFace",
                "status": "corrected",
                "contrast": f"{arm_a} minus {arm_b}",
                "arm_a_mean_AUC": fmt(values.mean()),
                "arm_b_mean_AUC": fmt(control_values.mean()),
                "paired_delta": fmt(result["mean_difference"]),
                "paired_CI95_low": fmt(result["ci95"][0]),
                "paired_CI95_high": fmt(result["ci95"][1]),
                "paired_t": fmt(result["t"]),
                "raw_p": fmt(result["p"]),
                "inference_note": note,
                "historical_TSKinFace_VQC": "No matching artifact found.",
                "source": "vqc_tskinface_corrected.json",
            }
        )
    write_csv("04_standard_vqc_vs_baselines.csv", vqc_rows)

    # Corrected TSKinFace quantum-vs-classical comparisons, by protocol.
    quantum_control_rows = []
    for backbone in BACKBONES:
        rows = corrected_amp[backbone]
        for quantum_arm in ("amp-expectation", "amp-fidelity"):
            quantum = np.asarray(rows[quantum_arm]["per_fold"])
            for control_arm in ("control", *WIDTH_ARMS):
                classical = np.asarray(rows[control_arm]["per_fold"])
                result = paired(quantum, classical)
                quantum_control_rows.append(
                    {
                        "dataset": "TSKinFace",
                        "backbone": backbone,
                        "method": quantum_arm,
                        "classical_control": control_arm,
                        "quantum_mean_AUC": fmt(quantum.mean()),
                        "control_mean_AUC": fmt(classical.mean()),
                        "paired_delta": fmt(result["mean_difference"]),
                        "CI95_low": fmt(result["ci95"][0]),
                        "CI95_high": fmt(result["ci95"][1]),
                        "t": fmt(result["t"]),
                        "raw_p": fmt(result["p"]),
                        "p_adjustment": (
                            "See 02_amplitude_vs_width_controls.csv for 32-test "
                            "Holm/BH adjustment; local control rows descriptive."
                        ),
                        "status": "corrected",
                        "source": amp_source_names[backbone]["TSKinFace"],
                    }
                )
    for arm in ("vqc-expectation", "vqc-fidelity"):
        quantum = np.asarray(corrected_vqc[arm]["per_fold"])
        classical = vqc_control
        result = paired(quantum, classical)
        quantum_control_rows.append(
            {
                "dataset": "TSKinFace",
                "backbone": "FaceNet",
                "method": f"standard {arm}",
                "classical_control": "capacity-matched VQC control",
                "quantum_mean_AUC": fmt(quantum.mean()),
                "control_mean_AUC": fmt(classical.mean()),
                "paired_delta": fmt(result["mean_difference"]),
                "CI95_low": fmt(result["ci95"][0]),
                "CI95_high": fmt(result["ci95"][1]),
                "t": fmt(result["t"]),
                "raw_p": fmt(result["p"]),
                "p_adjustment": "Not included in the amplitude-readout 16/32 test families.",
                "status": "corrected",
                "source": "vqc_tskinface_corrected.json",
            }
        )
    # Set-density comparison has summary means only in the two-backbone runner;
    # do not invent paired fold p-values for it.
    for backbone in BACKBONES:
        set_mean = new_backbone[backbone]["set"]["TSKinFace"]["mean"]
        density_mean = new_backbone[backbone]["set+density"]["TSKinFace"]["mean"]
        quantum_control_rows.append(
            {
                "dataset": "TSKinFace",
                "backbone": backbone,
                "method": "set+density",
                "classical_control": "set-level features",
                "quantum_mean_AUC": fmt(density_mean),
                "control_mean_AUC": fmt(set_mean),
                "paired_delta": fmt(density_mean - set_mean),
                "CI95_low": "NA",
                "CI95_high": "NA",
                "t": "NA",
                "raw_p": "NA",
                "p_adjustment": "Not calculable: corrected backbone artifact omits fold values.",
                "status": "corrected",
                "source": "backbone_comparison_tskinface_corrected.json",
            }
        )
    write_csv("05_quantum_inspired_vs_classical_controls.csv", quantum_control_rows)

    # Set-level aggregation comparison by dataset and backbone.
    set_rows = []
    for backbone in BACKBONES:
        for corpus in CORPORA:
            old_single = old_backbone[backbone]["single"][corpus]["mean"]
            old_set = old_backbone[backbone]["set"][corpus]["mean"]
            new_single = new_backbone[backbone]["single"][corpus]["mean"]
            new_set = new_backbone[backbone]["set"][corpus]["mean"]
            set_rows.append(
                {
                    "backbone": backbone,
                    "corpus": corpus,
                    "status": (
                        "corrected TSKinFace"
                        if corpus == "TSKinFace"
                        else "rerun; non-TSKinFace context"
                    ),
                    "historical_single_AUC": fmt(old_single),
                    "historical_set_AUC": fmt(old_set),
                    "historical_set_gain": fmt(old_set - old_single),
                    "corrected_single_AUC": fmt(new_single),
                    "corrected_set_AUC": fmt(new_set),
                    "corrected_set_gain": fmt(new_set - new_single),
                    "change_in_set_gain": fmt((new_set - new_single) - (old_set - old_single)),
                    "fold_p_value": "NA (fold-level scores not stored)",
                    "source_historical": "backbone_comparison.json",
                    "source_corrected": "backbone_comparison_tskinface_corrected.json",
                }
            )
    write_csv("06_set_level_aggregation.csv", set_rows)

    # Dataset-wise baseline and backbone performance.
    performance_rows = []
    for corpus in CORPORA:
        old = historical_kfold[corpus]
        new = corrected_kfold[corpus]
        row = {
            "corpus": corpus,
            "status": "corrected" if corpus == "TSKinFace" else "rerun_on_corrected_runner",
            "historical_kfold_accuracy_pct": fmt(old["accuracy_mean"], 3),
            "historical_kfold_accuracy_CI95_halfwidth_pct": fmt(old["accuracy_ci95"], 3),
            "historical_kfold_AUC": fmt(old["roc_auc_mean"]),
            "historical_kfold_AUC_SD": fmt(old["roc_auc_sd"]),
            "rerun_kfold_accuracy_pct": fmt(new["accuracy_mean"], 3),
            "rerun_kfold_accuracy_CI95_halfwidth_pct": fmt(new["accuracy_ci95"], 3),
            "rerun_kfold_AUC": fmt(new["roc_auc_mean"]),
            "rerun_kfold_AUC_SD": fmt(new["roc_auc_sd"]),
            "rerun_minus_historical_AUC": fmt(
                new["roc_auc_mean"] - old["roc_auc_mean"]
            ),
        }
        for backbone in BACKBONES:
            row[f"{backbone}_historical_single_AUC"] = fmt(
                old_backbone[backbone]["single"][corpus]["mean"]
            )
            row[f"{backbone}_historical_set_AUC"] = fmt(
                old_backbone[backbone]["set"][corpus]["mean"]
            )
            row[f"{backbone}_corrected_single_AUC"] = fmt(
                new_backbone[backbone]["single"][corpus]["mean"]
            )
            row[f"{backbone}_corrected_set_AUC"] = fmt(
                new_backbone[backbone]["set"][corpus]["mean"]
            )
        performance_rows.append(row)
    write_csv("07_dataset_and_backbone_performance.csv", performance_rows)

    # Keep a compact table of the nine manuscript formulations and distinguish
    # historical manuscript reports from corrected TSKinFace evidence.
    manuscript = [
        (1, "SWAP fidelity (5 seeds)", "classical head", "p=0.945", "Not rerun"),
        (2, "SWAP fidelity (20 folds)", "classical head", "p=0.982", "Not rerun"),
        (3, "Density fidelity", "mean-pooling", "-8.7 AUC", "Not rerun"),
        (4, "Interference phase sweep", "convex mixture", "+0.1 AUC", "Not rerun"),
        (5, "Entanglement entropy", "cosine", "0.514 alone", "Not rerun"),
        (6, "POVM head", "capacity-matched", "-5.5 AUC", "Not rerun"),
        (7, "Purity regulariser", "decorrelation", "-0.9 AUC; p=0.047", "Not rerun"),
        (8, "Density fidelity (sets)", "set features", "At most 0.0006 AUC change", "Rerun available"),
        (9, "Phase sweep (triads)", "convex mixture", "-0.0006 AUC; p=0.147", "Not rerun"),
    ]
    formulation_rows = []
    for number, name, control, history, status in manuscript:
        if number == 8:
            corrected_text = (
                "FaceNet delta=-0.000193; ArcFace delta=+0.000172 "
                "(summary means; fold-level p unavailable)"
            )
        else:
            corrected_text = "No corrected TSKinFace result artifact"
        formulation_rows.append(
            {
                "formulation": number,
                "method": name,
                "classical_control": control,
                "historical_manuscript_report": history,
                "corrected_TSKinFace_status": status,
                "corrected_TSKinFace_result": corrected_text,
                "source": "paper/elsevier/quantum_negative.tex; corrected set-density from backbone_comparison_tskinface_corrected.json",
            }
        )
    write_csv("08_nine_formulations_scope.csv", formulation_rows)

    rendered_tables = [
        "# Final comparison tables",
        "",
        "CSV files in this folder contain the full-precision comparison fields; "
        "the rendered tables below use those same rows.",
        "",
        "## 1. Scalar versus vector readout",
        "",
        markdown_table(
            "01_scalar_vs_vector_readout.csv",
            [
                ("Backbone", "backbone"),
                ("Corpus", "corpus"),
                ("Data status", "TSKinFace_status"),
                ("Historical Δ", "historical_vector_minus_scalar"),
                ("Composite Δ", "current_vector_minus_scalar"),
                ("95% CI low", "paired_CI95_low"),
                ("95% CI high", "paired_CI95_high"),
                ("p", "raw_p"),
                ("Holm, 16", "readout_Holm_16"),
                ("BH, 16", "readout_BH_16"),
            ],
        ),
        "",
        "## 2. Corrected amplitude VQC versus width-matched controls",
        "",
        markdown_table(
            "02_amplitude_vs_width_controls.csv",
            [
                ("Backbone", "backbone"),
                ("Corpus", "corpus"),
                ("Control", "classical_control"),
                ("Quantum AUC", "quantum_mean_AUC"),
                ("Control AUC", "classical_mean_AUC"),
                ("Δ", "paired_delta"),
                ("95% CI low", "CI95_low"),
                ("95% CI high", "CI95_high"),
                ("p", "raw_p"),
                ("Holm, 32", "Holm_32_all_controls"),
                ("BH, 32", "BH_32_all_controls"),
            ],
        ),
        "",
        "## 3. Historical versus corrected TSKinFace",
        "",
        markdown_table(
            "03_historical_vs_corrected_tskinface.csv",
            [
                ("Evaluation", "evaluation"),
                ("Backbone", "backbone"),
                ("Measure", "measure"),
                ("Historical", "historical_mean"),
                ("Corrected", "corrected_mean"),
                ("Corrected − historical", "corrected_minus_historical"),
                ("95% CI low", "paired_CI95_low"),
                ("95% CI high", "paired_CI95_high"),
                ("Paired p", "nominal_p"),
                ("Pairing limitation", "pairing_status"),
            ],
        ),
        "",
        "## 4. Standard VQC versus classical baselines",
        "",
        markdown_table(
            "04_standard_vqc_vs_baselines.csv",
            [
                ("Contrast", "contrast"),
                ("Arm A AUC", "arm_a_mean_AUC"),
                ("Arm B AUC", "arm_b_mean_AUC"),
                ("Δ", "paired_delta"),
                ("95% CI low", "paired_CI95_low"),
                ("95% CI high", "paired_CI95_high"),
                ("p", "raw_p"),
                ("Inference note", "inference_note"),
            ],
        ),
        "",
        "## 5. Corrected quantum-inspired methods versus classical controls",
        "",
        markdown_table(
            "05_quantum_inspired_vs_classical_controls.csv",
            [
                ("Backbone", "backbone"),
                ("Method", "method"),
                ("Classical control", "classical_control"),
                ("Method AUC", "quantum_mean_AUC"),
                ("Control AUC", "control_mean_AUC"),
                ("Δ", "paired_delta"),
                ("95% CI low", "CI95_low"),
                ("95% CI high", "CI95_high"),
                ("p", "raw_p"),
            ],
        ),
        "",
        "## 6. Set-level aggregation",
        "",
        markdown_table(
            "06_set_level_aggregation.csv",
            [
                ("Backbone", "backbone"),
                ("Corpus", "corpus"),
                ("Status", "status"),
                ("Historical single", "historical_single_AUC"),
                ("Historical set", "historical_set_AUC"),
                ("Historical gain", "historical_set_gain"),
                ("Corrected single", "corrected_single_AUC"),
                ("Corrected set", "corrected_set_AUC"),
                ("Corrected gain", "corrected_set_gain"),
            ],
        ),
        "",
        "## 7. Dataset-wise and backbone-wise performance",
        "",
        markdown_table(
            "07_dataset_and_backbone_performance.csv",
            [
                ("Corpus", "corpus"),
                ("Status", "status"),
                ("Historical baseline AUC", "historical_kfold_AUC"),
                ("Rerun baseline AUC", "rerun_kfold_AUC"),
                ("FaceNet single", "FaceNet_corrected_single_AUC"),
                ("FaceNet set", "FaceNet_corrected_set_AUC"),
                ("ArcFace single", "ArcFace_corrected_single_AUC"),
                ("ArcFace set", "ArcFace_corrected_set_AUC"),
            ],
        ),
        "",
        "## 8. Historical formulation coverage",
        "",
        markdown_table(
            "08_nine_formulations_scope.csv",
            [
                ("#", "formulation"),
                ("Method", "method"),
                ("Classical control", "classical_control"),
                ("Historical report", "historical_manuscript_report"),
                ("Corrected TSKinFace status", "corrected_TSKinFace_status"),
                ("Corrected result", "corrected_TSKinFace_result"),
            ],
        ),
        "",
    ]
    (OUT / "tables.md").write_text("\n".join(rendered_tables), encoding="utf-8")

    # Check calculated grouped-k-fold summaries against artifact metadata.
    checks = []
    for name, result in (
        ("historical", historical_kfold),
        ("corrected", corrected_kfold),
    ):
        for corpus in CORPORA:
            rows = result[corpus]["per_fold"]
            auc = np.asarray([row["roc_auc"] for row in rows])
            accuracy = np.asarray([row["accuracy"] for row in rows])
            auc_mean, auc_sd = mean_sd(auc)
            acc_mean, acc_sd = mean_sd(accuracy)
            acc_ci = 1.96 * acc_sd / np.sqrt(len(accuracy))
            checks.append(
                {
                    "artifact": "kfold.json"
                    if name == "historical"
                    else "kfold_tskinface_corrected.json",
                    "corpus": corpus,
                    "recomputed_AUC_mean": auc_mean,
                    "stored_AUC_mean": result[corpus]["roc_auc_mean"],
                    "recomputed_AUC_SD": auc_sd,
                    "stored_AUC_SD": result[corpus]["roc_auc_sd"],
                    "recomputed_accuracy_mean": acc_mean,
                    "stored_accuracy_mean": result[corpus]["accuracy_mean"],
                    "recomputed_accuracy_CI95_halfwidth": acc_ci,
                    "stored_accuracy_CI95_halfwidth": result[corpus]["accuracy_ci95"],
                }
            )
            if not np.isclose(auc_mean, result[corpus]["roc_auc_mean"], atol=1e-12):
                raise ValueError(f"AUC summary mismatch: {name} {corpus}")
            if not np.isclose(auc_sd, result[corpus]["roc_auc_sd"], atol=1e-12):
                raise ValueError(f"AUC SD mismatch: {name} {corpus}")
            if not np.isclose(acc_ci, result[corpus]["accuracy_ci95"], atol=1e-9):
                raise ValueError(f"Accuracy CI mismatch: {name} {corpus}")

    readout_positive = sum(row["readout_positive"] for row in readout_rows)
    readout_raw_significant = sum(
        result["p"] < 0.05 for _, _, result in readout_tests
    )
    audit = {
        "provenance": provenance,
        "corrected_metadata_checks": corrected_metadata_checks,
        "fold_metric_checks": fold_metric_checks,
        "summary_checks": checks,
        "historical_composite_recalculation": historical_pooled_validation,
        "corrected_composite_readout": composite,
        "multiplicity": {
            "historical_reference": {
                "paper_compatible_family_16": {
                    "readout_holm_survivors": int(
                        sum(historical_holm16[i] <= 0.05 for i in range(8))
                    ),
                    "width_control_holm_survivors": int(
                        sum(historical_holm16[i] <= 0.05 for i in range(8, 16))
                    ),
                    "readout_BH_survivors": int(
                        sum(historical_bh16[i] <= 0.05 for i in range(8))
                    ),
                    "width_control_BH_survivors": int(
                        sum(historical_bh16[i] <= 0.05 for i in range(8, 16))
                    ),
                    "manuscript_reported_readout_BH_survivors": 6,
                    "manuscript_reported_width_control_BH_survivors": 2,
                    "manuscript_BH_count_matches_recalculation": False,
                },
                "all_width_controls_family_32": {
                    "readout_holm_survivors": int(
                        sum(historical_holm32[i] <= 0.05 for i in range(8))
                    ),
                    "width_control_holm_survivors": int(
                        sum(historical_holm32[i] <= 0.05 for i in range(8, 32))
                    ),
                    "readout_BH_survivors": int(
                        sum(historical_bh32[i] <= 0.05 for i in range(8))
                    ),
                    "width_control_BH_survivors": int(
                        sum(historical_bh32[i] <= 0.05 for i in range(8, 32))
                    ),
                },
            },
            "paper_compatible_family_16": {
                "composition": "8 vector-vs-scalar tests + 8 vector-vs-foldwise-best width-control tests",
                "readout_holm_survivors": int(
                    sum(holm16[i] <= 0.05 for i in range(8))
                ),
                "width_control_holm_survivors": int(
                    sum(holm16[i] <= 0.05 for i in range(8, 16))
                ),
                "readout_BH_survivors": int(
                    sum(bh16[i] <= 0.05 for i in range(8))
                ),
                "width_control_BH_survivors": int(
                    sum(bh16[i] <= 0.05 for i in range(8, 16))
                ),
                "note": (
                    "The fold-wise maximum selects the strongest observed control "
                    "within each fold; this is optimistic and should be interpreted "
                    "as a compatibility check, not an unbiased control-selection test."
                ),
            },
            "all_width_controls_family_32": {
                "composition": "8 vector-vs-scalar tests + 24 vector-vs-individual-width-control tests",
                "readout_holm_survivors": int(
                    sum(holm32[i] <= 0.05 for i in range(8))
                ),
                "width_control_holm_survivors": int(
                    sum(holm32[i] <= 0.05 for i in range(8, 32))
                ),
                "readout_BH_survivors": int(
                    sum(bh32[i] <= 0.05 for i in range(8))
                ),
                "width_control_BH_survivors": int(
                    sum(bh32[i] <= 0.05 for i in range(8, 32))
                ),
                "method": "Two-sided paired t-test p-values adjusted over all 32 tests.",
            },
            "historical_BH_note": (
                "The LaTeX manuscript reports 6/8 readout and 2/8 width-control "
                "BH-FDR survivors. Recalculation from the 16 stored raw p-values "
                "with standard Benjamini-Hochberg gives 7/8 and 3/8, respectively. "
                "Historical Holm counts (5/8 and 1/8) do reproduce."
            ),
        },
        "eight_cell_readout_conclusion": {
            "positive_cells": readout_positive,
            "total_cells": 8,
            "unadjusted_p_below_0_05": readout_raw_significant,
            "all_positive": readout_positive == 8,
            "interpretation": (
                "Direction remains positive in all eight backbone-corpus cells. "
                "Not all per-cell tests remain significant: corrected ArcFace "
                "TSKinFace is positive but its nominal paired p-value exceeds 0.05."
            ),
        },
        "historical_vs_corrected_TSKinFace": tskinface_comparison,
        "notes": [
            "The historical setlevel.json reports a mean identity set size of 169.03 for TSKinFace, although each identity has one image. Treat its set-level comparison as invalid/unreliable; the independent backbone_comparison.json instead reports identical old single and set AUC.",
            "The corrected backbone comparison artifact stores means and SDs but not fold-level scores, so no paired p-value or CI is calculated for its set-density rows.",
            "Corrected set-level and backbone-comparison artifacts store summary statistics only; their fold-level CIs/SDs cannot be independently recomputed from those files. Corrected amplitude and standard-VQC ROC-AUC means/SDs were verified against all saved fold scores; amplitude per-fold accuracy scores were not saved.",
            "No historical TSKinFace standard-VQC result artifact was found; standard VQC corrected values are not compared to an invented historical estimate.",
            "Historical manuscript formulations 1-7 and 9 are summarized as reported in the LaTeX table only; corrected TSKinFace results are unavailable for those formulations.",
            "For the corrected composite, historical folds remain in place for FIW and both KinFaceW datasets. This is a mixed-provenance summary rather than a full rerun of 40 folds.",
        ],
    }
    save_json("statistical_audit.json", audit)

    # The report is intentionally generated from the same machine-readable
    # audit values, to keep all headline claims synchronized.
    facenet = composite["FaceNet"]
    arcface = composite["ArcFace"]
    tsk_readout = {
        row["backbone"]: row
        for row in readout_rows
        if row["corpus"] == "TSKinFace"
    }
    report = f"""# Corrected TSKinFace statistical audit

This audit independently recomputes paired tests from the saved fold arrays and
stores comparison tables in `tables/`. It does not modify the source results or
manuscript. Re-run it from the repository root with:

```powershell
.\\.venv\\Scripts\\python.exe results\\honest\\tskinface_statistical_audit\\recompute.py
```

## Executive findings

- Corrected grouped-k-fold TSKinFace baseline: ROC-AUC
  **{corrected_kfold['TSKinFace']['roc_auc_mean']:.4f}** (fold SD
  {corrected_kfold['TSKinFace']['roc_auc_sd']:.4f}), accuracy
  **{corrected_kfold['TSKinFace']['accuracy_mean']:.2f}% ±
  {corrected_kfold['TSKinFace']['accuracy_ci95']:.2f}** (the runner's 1.96-SE
  interval). Historical values were ROC-AUC
  {historical_kfold['TSKinFace']['roc_auc_mean']:.4f} and accuracy
  {historical_kfold['TSKinFace']['accuracy_mean']:.2f}%.
- The vector-minus-scalar amplitude readout effect remains positive in
  **{readout_positive}/8** backbone-corpus cells. Only
  **{readout_raw_significant}/8** cell-level paired tests are nominally below
  0.05; corrected ArcFace TSKinFace is positive
  ({float(tsk_readout['ArcFace']['current_vector_minus_scalar']):+.4f}) but
  not individually significant (p={float(tsk_readout['ArcFace']['raw_p']):.4f}).
- Corrected composite paired-fold readout deltas (20 folds/backbone) are
  FaceNet **{facenet['fold_level_paired_test']['mean_difference']:+.4f}**
  (95% CI [{facenet['fold_level_paired_test']['ci95'][0]:+.4f},
  {facenet['fold_level_paired_test']['ci95'][1]:+.4f}], p=
  {facenet['fold_level_paired_test']['p']:.3g}) and ArcFace
  **{arcface['fold_level_paired_test']['mean_difference']:+.4f}**
  (95% CI [{arcface['fold_level_paired_test']['ci95'][0]:+.4f},
  {arcface['fold_level_paired_test']['ci95'][1]:+.4f}], p=
  {arcface['fold_level_paired_test']['p']:.3g}).
- At the more conservative corpus unit (n=4), the mean effects are
  {facenet['corpus_level_one_sample_test']['mean_difference']:+.4f}
  (p={facenet['corpus_level_one_sample_test']['p']:.4f}) for FaceNet and
  {arcface['corpus_level_one_sample_test']['mean_difference']:+.4f}
  (p={arcface['corpus_level_one_sample_test']['p']:.4f}) for ArcFace.
- Holm/BH counts are in `statistical_audit.json`. For the paper-compatible
  16-test family, the readout cells surviving are
  {audit['multiplicity']['paper_compatible_family_16']['readout_holm_survivors']}/8
  by Holm and
  {audit['multiplicity']['paper_compatible_family_16']['readout_BH_survivors']}/8
  by BH-FDR. Historically, Holm reproduces the manuscript's 5/8 readout and
  1/8 width-control counts, but standard BH-FDR recomputes to 7/8 and 3/8,
  not the manuscript's reported 6/8 and 2/8. A separate 32-test analysis
  compares against every individual
  width-matched control rather than selecting a winner.

## Historical versus corrected TSKinFace

See `tables/03_historical_vs_corrected_tskinface.csv`. The corrected
k-fold baseline is lower than the historical score. The readout gap also
shrinks: historical FaceNet and ArcFace vector-minus-scalar effects were
{old_readout_summary['per_cell']['FaceNet']['TSKinFace']['readout_delta']:+.4f}
and
{old_readout_summary['per_cell']['ArcFace']['TSKinFace']['readout_delta']:+.4f};
the corrected effects are
{float(tsk_readout['FaceNet']['current_vector_minus_scalar']):+.4f} and
{float(tsk_readout['ArcFace']['current_vector_minus_scalar']):+.4f}.

Paired historical-versus-corrected p-values in the comparison table assume
fold-index alignment. Historical files lack fold manifests and source commits,
so those comparisons are explicitly exploratory.

## Requested comparison tables

The rendered tables are in [`tables.md`](tables.md); CSVs are in [`tables/`](tables/).

1. `tables/01_scalar_vs_vector_readout.csv` — historical and current values,
   paired CIs/p-values, and multiplicity adjustments for all eight cells.
2. `tables/02_amplitude_vs_width_controls.csv` — every width-matched control,
   not only the selected best control.
3. `tables/03_historical_vs_corrected_tskinface.csv` — direct historical /
   corrected comparisons with paired-fold caveats.
4. `tables/04_standard_vqc_vs_baselines.csv` — corrected standard VQC with
   paired tests against its capacity-matched control, the alternate VQC
   readout, and the global grouped-k-fold baseline. The latter uses matched
   held-out folds but a different, pooled-corpus training protocol.
5. `tables/05_quantum_inspired_vs_classical_controls.csv` — corrected
   amplitude, standard-VQC, and set-density comparisons.
6. `tables/06_set_level_aggregation.csv` — single versus set-level results for
   each corpus and backbone, with historical and corrected labels.
7. `tables/07_dataset_and_backbone_performance.csv` — grouped-k-fold dataset
   metrics and per-backbone single/set results.
8. `tables/08_nine_formulations_scope.csv` — manuscript-reported formulations
   and whether corrected TSKinFace evidence exists.

## Interpretation and limitations

Fold tests use paired t statistics and t-based 95% intervals. Folds partition
the same corpora and are not independent population samples. The corpus-level
results are therefore shown separately. The composite substitutes corrected
TSKinFace folds while retaining historical folds for the other three corpora;
it is not a full corrected 40-fold rerun.

The historical `setlevel.json` has an implausible TSKinFace mean set size of
169.03 despite one image per identity. Its old TSKinFace set-level results are
not treated as reliable; `backbone_comparison.json` is reported separately.
Corrected density-fidelity summary artifacts omit fold scores, so no inferential
p-value is invented for those comparisons. Corrected set-level and backbone
summary means/SDs cannot be rederived from per-fold data because the runs did
not save fold-level values. All corrected amplitude and standard-VQC ROC-AUC
means/SDs do match their saved fold arrays; amplitude per-fold accuracy was
not saved. No historical TSKinFace standard VQC result was found.

The nine-formulation table in the manuscript is preserved as historical context.
Only the set-density formulation among formulations 1--9 has a corrected
TSKinFace result in the available reruns; the other formulations need their own
reruns before corrected TSKinFace claims can be made.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(f"Wrote audit to {OUT}")
    print(
        "Readout positive cells: "
        f"{readout_positive}/8; nominal p < .05: {readout_raw_significant}/8"
    )
    for backbone in BACKBONES:
        result = composite[backbone]["fold_level_paired_test"]
        corpus = composite[backbone]["corpus_level_one_sample_test"]
        print(
            f"{backbone}: fold delta={result['mean_difference']:+.6f}, "
            f"CI={result['ci95']}, p={result['p']:.6g}; "
            f"corpus p={corpus['p']:.6g}"
        )
    print(
        "Multiplicity Holm/BH (16): readout="
        f"{audit['multiplicity']['paper_compatible_family_16']['readout_holm_survivors']}/8/"
        f"{audit['multiplicity']['paper_compatible_family_16']['readout_BH_survivors']}/8"
    )


if __name__ == "__main__":
    main()
