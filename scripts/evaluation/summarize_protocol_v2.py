# -*- coding: utf-8 -*-
"""Turn results/protocol_v2 into SUMMARY.md and summary.json.

Reads only saved records and scores, and tolerates a partly finished run: any
comparison whose folds are not all present is listed as incomplete.
"""
import glob
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.fair_protocol import bootstrap_auc_delta, pool_fold_scores, t_interval

OUT = os.path.join(ROOT, "results", "protocol_v2")
CORPORA = ("FIW", "KinFaceW-I", "KinFaceW-II", "TSKinFace")
BACKBONES = ("facenet", "arcface")
FOLDS = 5
N_BOOT = 2000
INCOMPLETE_COMPARISONS = {}
TRAINED_ARMS = {"fair", "fair_colour_trained", "hybrid_q0", "hybrid_q1",
                "hybrid_paper_q1"}


def load(name):
    path = os.path.join(OUT, name + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def negative_seed_offsets():
    offsets = set()
    for path in glob.glob(os.path.join(OUT, "*.jsonl")):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if "negative_seed_offset" in record:
                    offsets.add(int(record["negative_seed_offset"]))
    return sorted(offsets)


def by_arm(records, metric):
    """{(arm, corpus): [fold means over seeds]}."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in records:
        if metric in r and "corpus" in r:
            cells[(r["arm"], r["corpus"])][r["fold"]].append(r[metric])
    return {key: [float(np.mean(v)) for _f, v in sorted(folds.items())]
            for key, folds in cells.items()}


def training_run_summary(records):
    """Deduplicate model-level validation outcomes repeated for each corpus."""
    runs = {}
    fields = ("val_auc", "val_cosine_auc", "train_auc", "train_cosine_auc",
              "failed", "epochs")
    for record in records:
        if record.get("arm") not in TRAINED_ARMS:
            continue
        key = (record["arm"], record["fold"], record["seed"])
        current = runs.setdefault(key, {})
        for field in fields:
            if field not in record:
                continue
            value = record[field]
            if field in current and current[field] != value:
                raise ValueError(f"Inconsistent {field} across corpora for run {key}")
            current[field] = value
    return [{"arm": key[0], "fold": key[1], "seed": key[2], **runs[key]}
            for key in sorted(runs)]


def fmt(values):
    if not values:
        return "-"
    mean, half = t_interval(values)
    return f"{mean:.3f}" if np.isnan(half) else f"{mean:.3f} ± {half:.3f}"


def load_folds(experiment, backbone, arm, corpus, view):
    """Per-fold dicts for pooling, or None when data is missing or inconsistent."""
    score_key = {"matched": "matched", "colour": "colour_matched"}[view]
    y_key, groups_a_key, groups_b_key = (
        ("y", "groups_a", "groups_b") if view == "matched"
        else ("y_colour", "groups_colour_a", "groups_colour_b")
    )
    folds = []
    for fold in range(FOLDS):
        paths = sorted(glob.glob(os.path.join(
            OUT, "scores", glob.escape(f"{experiment}_{backbone}_{arm}_{corpus}_f{fold}_") + "s*.npz")))
        if not paths:
            INCOMPLETE_COMPARISONS[(experiment, backbone, arm, corpus, view, fold)] = (
                "no score files")
            return None
        required = (score_key, y_key, groups_a_key, groups_b_key)
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if any(key not in archive.files for key in required):
                    INCOMPLETE_COMPARISONS[
                        (experiment, backbone, arm, corpus, view, fold)
                    ] = f"legacy or incomplete score schema: {os.path.basename(path)}"
                    return None
                arrays.append({key: archive[key] for key in archive.files})

        base = arrays[0]
        base_has_index = "index" in base
        expected = (base[y_key], base[groups_a_key], base[groups_b_key])
        if any(value.ndim != 1 for value in expected):
            INCOMPLETE_COMPARISONS[
                (experiment, backbone, arm, corpus, view, fold)
            ] = "labels or endpoint-family arrays are not one-dimensional"
            return None
        if len({len(value) for value in expected}) != 1:
            INCOMPLETE_COMPARISONS[
                (experiment, backbone, arm, corpus, view, fold)
            ] = "labels and endpoint-family arrays have different lengths"
            return None
        if base_has_index:
            index = base["index"]
            if (index.ndim != 1 or len(index) != len(base[y_key])
                    or len(np.unique(index)) != len(index)):
                INCOMPLETE_COMPARISONS[
                    (experiment, backbone, arm, corpus, view, fold)
                ] = "pair indices are invalid"
                return None

        scores = []
        for path, values in zip(paths, arrays):
            score = values[score_key]
            if score.ndim != 1 or len(score) != len(base[y_key]) or not np.isfinite(score).all():
                INCOMPLETE_COMPARISONS[
                    (experiment, backbone, arm, corpus, view, fold)
                ] = f"invalid score vector: {os.path.basename(path)}"
                return None
            if ("index" in values) != base_has_index:
                INCOMPLETE_COMPARISONS[
                    (experiment, backbone, arm, corpus, view, fold)
                ] = "seed score files disagree about pair indices"
                return None
            if any(not np.array_equal(values[key], base[key])
                   for key in (y_key, groups_a_key, groups_b_key)):
                INCOMPLETE_COMPARISONS[
                    (experiment, backbone, arm, corpus, view, fold)
                ] = f"seed score files disagree on pair labels or family IDs: {os.path.basename(path)}"
                return None
            if base_has_index and not np.array_equal(values["index"], base["index"]):
                INCOMPLETE_COMPARISONS[
                    (experiment, backbone, arm, corpus, view, fold)
                ] = f"seed score files disagree on pair indices: {os.path.basename(path)}"
                return None
            scores.append(score)

        entry = {"scores": scores, "y": base[y_key],
                 "groups_a": base[groups_a_key], "groups_b": base[groups_b_key]}
        if base_has_index:
            entry["index"] = base["index"]
        folds.append(entry)
    return folds


def align(first, second):
    """Keep only pairs both arms scored; needed once shared photos are removed."""
    out_a, out_b = [], []
    for a, b in zip(first, second):
        if ("index" in a) != ("index" in b):
            raise ValueError("Arms disagree about whether pair indices are available")
        if "index" not in a:
            if (len(a["y"]) != len(b["y"])
                    or not np.array_equal(a["y"], b["y"])
                    or not np.array_equal(a["groups_a"], b["groups_a"])
                    or not np.array_equal(a["groups_b"], b["groups_b"])):
                raise ValueError("Arms do not share the same held-out pairs")
            out_a.append(a)
            out_b.append(b)
            continue
        for fold in (a, b):
            if (np.any(np.diff(fold["index"]) <= 0)
                    or len(fold["index"]) != len(fold["y"])):
                raise ValueError("Pair indices must be unique, sorted, and aligned")
        common = np.intersect1d(a["index"], b["index"])
        if len(common) == 0:
            raise ValueError("Arms have no shared held-out pairs")
        pick = lambda f: {
            "scores": [s[np.searchsorted(f["index"], common)] for s in f["scores"]],
            "y": f["y"][np.searchsorted(f["index"], common)],
            "groups_a": f["groups_a"][np.searchsorted(f["index"], common)],
            "groups_b": f["groups_b"][np.searchsorted(f["index"], common)]}
        out_a.append(pick(a))
        out_b.append(pick(b))
    return out_a, out_b


def compare(experiment, backbone, arm_a, arm_b, corpus, view="matched"):
    first = load_folds(experiment, backbone, arm_a, corpus, view)
    second = load_folds(experiment, backbone, arm_b, corpus, view)
    if first is None or second is None:
        return None
    first, second = align(first, second)
    y, a, groups_a, groups_b = pool_fold_scores(first)
    y_b, b, groups_a_b, groups_b_b = pool_fold_scores(second)
    if not (np.array_equal(y, y_b) and np.array_equal(groups_a, groups_a_b)
            and np.array_equal(groups_b, groups_b_b)):
        raise ValueError(f"Arms do not share the same held-out pairs: {arm_a} vs {arm_b}")
    return bootstrap_auc_delta(y, a, b, groups_a, groups_b,
                               n_boot=N_BOOT, seed=0)


def delta_table(rows):
    lines = ["| Backbone | Corpus | Comparison | View | Δ AUC | 95% CI |",
             "|---|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['backbone']} | {r['corpus']} | {r['a']} − {r['b']} | {r['view']} | "
            f"{r['delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    return lines


def family(experiment_for, comparisons, corpora, views=("matched", "colour"), backbones=BACKBONES):
    rows = []
    for backbone in backbones:
        for corpus in corpora:
            for a, b in comparisons:
                for view in views:
                    out = compare(experiment_for, backbone, a, b, corpus, view)
                    if out is not None:
                        rows.append({"backbone": backbone, "corpus": corpus, "a": a, "b": b,
                                     "view": view, **out})
    return rows


def arm_table(records, arms, metrics, corpora=CORPORA):
    lines = ["| Corpus | Arm | " + " | ".join(m for m, _ in metrics) + " |",
             "|---|---|" + "---:|" * len(metrics)]
    cells = {key: by_arm(records, key) for _m, key in metrics}
    for corpus in corpora:
        for arm in arms:
            values = [cells[key].get((arm, corpus), []) for _m, key in metrics]
            if any(values):
                lines.append(f"| {corpus} | {arm} | " + " | ".join(fmt(v) for v in values) + " |")
    return lines


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=OUT,
                        help="results directory containing protocol JSONL and scores")
    OUT = os.path.abspath(parser.parse_args().input_dir)
    INCOMPLETE_COMPARISONS.clear()
    md = ["# Protocol v2 results", "",
          "Generated by `scripts/evaluation/summarize_protocol_v2.py`; plan in "
          "`docs/PROTOCOL_V2.md`. `±` is the Student-t half-width across folds and is "
          "descriptive. Δ rows show 95% intervals from a two-endpoint family "
          "pigeonhole bootstrap on within-fold ranks; no bootstrap p-values are reported. "
          "\"colour\" means negatives also matched on colour distance.", ""]
    data = {}
    offsets = negative_seed_offsets()
    data["negative_seed_offsets"] = offsets
    md += [f"Negative-sampling seed offset: {', '.join(map(str, offsets)) if offsets else 'unknown'}.",
           "Use a separate output directory for each offset; model seeds and family folds "
           "remain fixed.", ""]

    # ---- Q1, Q2, Q10
    for backbone in BACKBONES:
        records = load("reference_" + backbone)
        if not records:
            continue
        md += [f"## Reference, {backbone}", ""]
        md += arm_table(records, ("colour", "cosine", "lr", "fair", "fair_colour_trained",
                                  "hybrid_q0", "hybrid_q1", "hybrid_paper_q1"),
                        (("matched", "auc_matched"), ("colour-matched", "auc_colour_matched"),
                         ("old negatives", "auc_old"), ("accuracy % (matched)", "accuracy"),
                         ("FPR %", "fpr")))
        trained = training_run_summary(records)
        failed = [r for r in trained if r.get("failed")]
        failed_detail = [
            f"{r['arm']}/f{r['fold']}/s{r['seed']} "
            f"(validation {r.get('val_auc', float('nan')):.3f} vs cosine "
            f"{r.get('val_cosine_auc', float('nan')):.3f}; "
            + (f"train AUC {r['train_auc']:.3f}" if "train_auc" in r
               else "train AUC unavailable in this run")
            + ")"
            for r in failed
        ]
        md += ["", f"Unique trained model fits: {len(trained)}; flagged below the "
                   f"validation cosine baseline: {len(failed)} "
                   f"{'; '.join(failed_detail) if failed_detail else ''}", ""]
        data[f"training_{backbone}"] = {"runs": trained, "flagged": failed}
    reference_rows = family("reference", [("fair", "cosine"), ("fair_colour_trained", "cosine"),
                                          ("lr", "cosine"), ("hybrid_paper_q1", "cosine")], CORPORA)
    if reference_rows:
        md += ["## Reference model against untrained cosine", ""] + delta_table(reference_rows) + [""]
    data["reference"] = reference_rows

    # ---- Stability
    rows = []
    for backbone in BACKBONES:
        records = [r for r in load("reference_" + backbone) if r["arm"] == "fair"]
        for corpus in CORPORA:
            subset = [r for r in records if r["corpus"] == corpus]
            if len(subset) < 2:
                continue
            per_fold, per_seed = defaultdict(list), defaultdict(list)
            for r in subset:
                per_fold[r["fold"]].append(r["accuracy"])
                per_seed[r["seed"]].append(r["accuracy"])
            fold_means = [np.mean(v) for v in per_fold.values()]
            within_fold = [max(v) - min(v) for v in per_fold.values()]
            rows.append(f"| {backbone} | {corpus} | {max(fold_means) - min(fold_means):.1f} | "
                        f"{np.mean(within_fold):.1f} | {min(r['accuracy'] for r in subset):.1f}–"
                        f"{max(r['accuracy'] for r in subset):.1f} |")
    if rows:
        md += ["## Stability of the fair model (accuracy points)", "",
               "| Backbone | Corpus | Range across folds | Mean range across seeds | All runs |",
               "|---|---|---:|---:|---:|"] + rows + [""]

    # ---- Q3, Q4
    readout_scalar = [("expectation", "fidelity"), ("expectation", "fidelity_log"),
                      ("expectation", "swap")]
    readout_width = [("expectation", "width_linear"), ("expectation", "width_mlp"),
                     ("expectation", "width_random")]
    for backbone in BACKBONES:
        records = load("readout_" + backbone)
        if records:
            md += [f"## Readout, {backbone}", ""]
            md += arm_table(records, ("expectation", "fidelity", "fidelity_log", "swap", "control",
                                      "width_linear", "width_mlp", "width_random"),
                            (("matched", "auc_matched"), ("colour-matched", "auc_colour_matched"),
                             ("cosine", "cosine"), ("colour only", "colour_only")))
            md.append("")
    for name, comparisons in (("Vector against scalar readouts", readout_scalar),
                              ("Circuit against width-matched classical controls", readout_width)):
        rows = family("readout", comparisons, CORPORA)
        if rows:
            md += [f"## {name}", ""] + delta_table(rows) + [""]
        data[name] = rows

    # ---- Q5
    rows = []
    rows += family("reference", [("hybrid_q1", "hybrid_q0")], CORPORA)
    rows += family("setlevel", [("set+density", "set")], CORPORA, views=("matched",))
    rows += family("triadic", [("triad+phase_gain", "triad"), ("triad+phase_all", "triad")],
                   ("TSKinFace",))
    rows += family("formulations", [("f5_entropy", "f5_control"), ("f6_povm", "f6_control"),
                                    ("f7_purity", "f7_control")], CORPORA, views=("matched",))
    if rows:
        md += ["## Quantum-inspired formulations against their controls", ""]
        md += delta_table(rows) + [""]
    data["formulations"] = rows
    for backbone in BACKBONES:
        f1 = [r for r in load("formulations_" + backbone) if r["arm"].startswith("f1_")]
        on = {r["seed"]: r["auc_matched"] for r in f1 if r["arm"].endswith("q1")}
        off = {r["seed"]: r["auc_matched"] for r in f1 if r["arm"].endswith("q0")}
        shared = sorted(set(on) & set(off))
        if shared:
            md += [f"Formulation 1, {backbone}, FIW holdouts: on − off = "
                   f"{fmt([on[s] - off[s] for s in shared])} over {len(shared)} seeds.", ""]

    # ---- Q6
    for backbone in BACKBONES:
        records = load("setlevel_" + backbone)
        if records:
            md += [f"## Set-level, {backbone}", ""]
            md += arm_table(records, ("single", "set", "set+density", "set_noshared",
                                      "single_noshared"),
                            (("matched", "auc_matched"),)) + [""]
    rows = family("setlevel", [("set", "single"),
                               ("set_noshared", "single_noshared")], ("FIW",),
                  views=("matched",))
    if rows:
        md += ["## Set-level gain on FIW", "",
               "`set_noshared` is compared with `single_noshared` on the same retained "
               "photo-pair rows. Shared-source query pairs are excluded from both arms.", ""]
        md += delta_table(rows) + [""]
    data["setlevel"] = rows

    # ---- Q7
    for backbone in BACKBONES:
        records = load("triadic_" + backbone)
        if records:
            md += [f"## Triadic, {backbone}", ""]
            md += arm_table(records, ("best_parent", "triad", "triad+phase_gain", "triad+phase_all"),
                            (("matched", "auc_matched"), ("colour-matched", "auc_colour_matched")),
                            corpora=("TSKinFace",)) + [""]
    rows = family("triadic", [("triad", "best_parent")], ("TSKinFace",))
    if rows:
        md += ["## Triadic gain", ""] + delta_table(rows) + [""]
    data["triadic"] = rows

    # ---- Q8
    records = load("rfiw_facenet")
    if records:
        cells = defaultdict(list)
        for r in records:
            cells[(r["train_on"], r["arm"], r["view"])].append(r)
        md += ["## Official RFIW Track-I split (FaceNet)", "",
               "Uses the 74% of official test pairs whose faces are available. It is not "
               "comparable to the full leaderboard.", "",
               "| Trained on | Arm | Test view | AUC | Accuracy % | Seeds |",
               "|---|---|---|---:|---:|---:|"]
        for (train_on, arm, view), group in sorted(cells.items()):
            accuracy = [r["accuracy"] for r in group if "accuracy" in r]
            md.append(f"| {train_on} | {arm} | {view} | {fmt([r['auc'] for r in group])} | "
                      f"{fmt(accuracy)} | {len(group)} |")
        md.append("")
        data["rfiw"] = [{k: r.get(k) for k in ("arm", "train_on", "view", "seed", "auc", "accuracy", "fpr")}
                        for r in records]

    # ---- Q9
    records = load("cost")
    if records:
        base = next((r for r in records if r["arm"] == "fair"), None)
        md += ["## Cost (GPU, batch 256, 3 warm-ups, 30 repeats, median)", "",
               "| Model | Parameters | Inference pairs/s | Training pairs/s | Inference cost vs fair | Training cost vs fair |",
               "|---|---:|---:|---:|---:|---:|"]
        for r in records:
            md.append(f"| {r['arm']} | {r['parameters']:,} | {r['inference_pairs_per_s']:,.0f} | "
                      f"{r['train_pairs_per_s']:,.0f} | "
                      f"{base['inference_pairs_per_s'] / r['inference_pairs_per_s']:.1f}× | "
                      f"{base['train_pairs_per_s'] / r['train_pairs_per_s']:.1f}× |")
        md.append("")
        data["cost"] = [{k: v for k, v in r.items() if k != "provenance"} for r in records]

    incomplete = [
        {"experiment": key[0], "backbone": key[1], "arm": key[2],
         "corpus": key[3], "view": key[4], "fold": key[5], "reason": reason}
        for key, reason in sorted(INCOMPLETE_COMPARISONS.items())
    ]
    if incomplete:
        md += ["## Incomplete paired comparisons", "",
               "These comparisons were omitted because one or more score folds were "
               "missing, used the legacy schema, or failed pair-alignment validation.", "",
               "| Experiment | Backbone | Arm | Corpus | View | Fold | Reason |",
               "|---|---|---|---|---|---:|---|"]
        md += [f"| {r['experiment']} | {r['backbone']} | {r['arm']} | {r['corpus']} | "
               f"{r['view']} | {r['fold']} | {r['reason']} |" for r in incomplete]
    data["incomplete_comparisons"] = incomplete

    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=1)
    print("wrote SUMMARY.md and summary.json;", len(md), "lines")


if __name__ == "__main__":
    main()
