"""Audit RFIW Track-I test-pair coverage by kin label and relation.

The audit uses only the official labelled CSV, local test-face files, and the
project's own RFIW embedding cache. It reports descriptive coverage rates and
a two-endpoint family-cluster bootstrap interval, not a hypothesis-test p-value.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coverage_records(csv_path, faces_root, cache_path):
    with open(cache_path, "rb") as stream:
        cache = pickle.load(stream)
    cached_paths = {norm(path) for path in cache}
    records = []
    with open(csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative_paths = [row["p1"].strip(), row["p2"].strip()]
            paths = [os.path.join(faces_root, *path.split("/"))
                     for path in relative_paths]
            covered = all(os.path.isfile(path) and norm(path) in cached_paths
                          for path in paths)
            records.append({
                "label": int(row["labels"]),
                "relation": row["ptype"].strip().lower(),
                "family_a": relative_paths[0].split("/")[0],
                "family_b": relative_paths[1].split("/")[0],
                "covered": covered,
            })
    return records


def rate_table(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record["covered"])
    return {
        str(key): {
            "total": len(values),
            "covered": int(sum(values)),
            "coverage_rate": float(np.mean(values)),
        }
        for key, values in sorted(groups.items())
    }


def bootstrap_coverage_difference(records, n_boot=2000, seed=42):
    """Kin-minus-nonkin coverage difference with two-endpoint family weights."""
    labels = np.asarray([record["label"] for record in records])
    covered = np.asarray([record["covered"] for record in records], dtype=float)
    if len(records) == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("RFIW test pairs must be non-empty and use binary labels 0/1")
    families = np.asarray([record["family_a"] for record in records]
                          + [record["family_b"] for record in records])
    keys, inverse = np.unique(families, return_inverse=True)
    index_a, index_b = inverse[:len(records)], inverse[len(records):]
    same_family = index_a == index_b
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError("RFIW test pairs must contain kin and non-kin labels")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    positive_rate = float(covered[positive].mean())
    negative_rate = float(covered[negative].mean())
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        counts = np.bincount(
            rng.integers(0, len(keys), len(keys)), minlength=len(keys))
        weights = counts[index_a] * counts[index_b]
        weights[same_family] = counts[index_a[same_family]]
        positive_weight = weights * positive
        negative_weight = weights * negative
        if positive_weight.sum() == 0 or negative_weight.sum() == 0:
            continue
        differences.append(
            np.sum(positive_weight * covered) / positive_weight.sum()
            - np.sum(negative_weight * covered) / negative_weight.sum())
    if len(differences) < max(100, n_boot // 2):
        raise RuntimeError(
            f"Only {len(differences)}/{n_boot} valid family-bootstrap samples")
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "kin_coverage_rate": positive_rate,
        "nonkin_coverage_rate": negative_rate,
        "difference_percentage_points": 100 * (positive_rate - negative_rate),
        "ci_low_percentage_points": float(100 * low),
        "ci_high_percentage_points": float(100 * high),
        "families": int(len(keys)),
        "valid_bootstrap_replicates": len(differences),
        "bootstrap": "two-endpoint family pigeonhole; percentile 95% CI",
    }


def audit(csv_path, faces_root, cache_path, n_boot=2000, seed=42):
    records = load_coverage_records(csv_path, faces_root, cache_path)
    covered_count = sum(record["covered"] for record in records)
    return {
        "split": "Official RFIW Track-I test",
        "total_pairs": len(records),
        "covered_pairs": int(covered_count),
        "uncovered_pairs": int(len(records) - covered_count),
        "overall_coverage_rate": float(covered_count / len(records)),
        "by_label": rate_table(records, "label"),
        "by_relation": rate_table(records, "relation"),
        "kin_vs_nonkin": bootstrap_coverage_difference(
            records, n_boot=n_boot, seed=seed),
        "provenance": {
            "test_pairs_csv_sha256": sha256(csv_path),
            "embedding_cache_sha256": sha256(cache_path),
        },
        "interpretation": (
            "Coverage differs by label in this test split. The available subset "
            "is therefore not a neutral random sample; results apply to the "
            "covered pairs only. Rates are descriptive and the interval accounts "
            "for repeated family endpoints, but not other sources of selection."
        ),
    }


def render_markdown(result):
    lines = [
        "# RFIW Track-I test coverage audit",
        "",
        result["interpretation"],
        "",
        f"Covered {result['covered_pairs']:,} of {result['total_pairs']:,} pairs "
        f"({result['overall_coverage_rate']:.1%}).",
        "",
        "| Label | Test pairs | Covered | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for label, name in (("1", "Kin"), ("0", "Non-kin")):
        row = result["by_label"][label]
        lines.append(
            f"| {name} | {row['total']:,} | {row['covered']:,} | "
            f"{row['coverage_rate']:.1%} |")
    difference = result["kin_vs_nonkin"]
    lines += [
        "",
        f"Kin-minus-non-kin coverage difference: "
        f"{difference['difference_percentage_points']:+.2f} percentage points "
        f"(95% family-cluster CI "
        f"[{difference['ci_low_percentage_points']:+.2f}, "
        f"{difference['ci_high_percentage_points']:+.2f}]).",
        "",
        "| Relation | Test pairs | Covered | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for relation, row in result["by_relation"].items():
        lines.append(
            f"| {relation} | {row['total']:,} | {row['covered']:,} | "
            f"{row['coverage_rate']:.1%} |")
    lines += [
        "",
        "The interval is a descriptive two-endpoint family-cluster percentile "
        "interval, not a hypothesis-test p-value.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(
        ROOT, "public", "rfiw_track1", "test-pairs.csv"))
    parser.add_argument("--faces", default=os.path.join(
        ROOT, "public", "rfiw_track1", "test-fids"))
    parser.add_argument("--cache", default=os.path.join(
        ROOT, "weights", "caches", "rfiw_track1_cache.pkl"))
    parser.add_argument("--output", default=os.path.join(
        ROOT, "results", "honest", "rfiw_test_coverage.json"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = audit(args.csv, args.faces, args.cache,
                   args.bootstrap_replicates, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    markdown_path = os.path.splitext(args.output)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as stream:
        stream.write(render_markdown(result))
    print(f"Wrote {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
