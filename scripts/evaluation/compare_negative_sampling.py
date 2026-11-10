"""Compare protocol-v2 summaries across independent negative-sampling draws.

The across-draw mean, SD and range are descriptive sensitivity statistics.
They are not hypothesis tests, and per-draw family-bootstrap intervals are
kept separate rather than pooled.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def load_run(input_dir):
    summary_path = os.path.join(input_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"summary not found: {summary_path}")
    with open(summary_path, encoding="utf-8") as stream:
        summary = json.load(stream)

    offsets = summary.get("negative_seed_offsets")
    if not isinstance(offsets, list) or len(offsets) != 1:
        raise ValueError(f"{input_dir} must summarize exactly one negative-seed offset")

    source_hashes = set()
    for name in os.listdir(input_dir):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("negative_seed_offset") != offsets[0]:
                    raise ValueError(
                        f"{name}:{line_number} has a different negative-seed offset")
                provenance = record.get("provenance", {})
                source_hash = provenance.get("source_sha256")
                if source_hash:
                    source_hashes.add(source_hash)
    if len(source_hashes) != 1:
        raise ValueError(
            f"{input_dir} has {len(source_hashes)} protocol source fingerprints; "
            "expected one")

    rows = {}
    for section, value in summary.items():
        candidates = value if isinstance(value, list) else (
            value.get("comparisons", []) if isinstance(value, dict) else [])
        if not isinstance(candidates, list):
            continue
        for row in candidates:
            if not isinstance(row, dict):
                continue
            required = ("backbone", "corpus", "a", "b", "view", "delta",
                        "ci_low", "ci_high")
            if not all(key in row for key in required):
                continue
            key = (section, row["backbone"], row["corpus"], row["a"],
                   row["b"], row["view"])
            if key in rows:
                raise ValueError(f"duplicate comparison in {input_dir}: {key}")
            rows[key] = {field: row[field] for field in required if field != "backbone"}
    return int(offsets[0]), next(iter(source_hashes)), rows


def compare_runs(input_dirs):
    if len(input_dirs) < 2:
        raise ValueError("provide at least two summarized runs with different negative draws")

    runs = [load_run(os.path.abspath(path)) for path in input_dirs]
    offsets = [run[0] for run in runs]
    if len(set(offsets)) != len(offsets):
        raise ValueError("each input directory must use a distinct negative-seed offset")
    fingerprints = {run[1] for run in runs}
    if len(fingerprints) != 1:
        raise ValueError("runs used different protocol source fingerprints")

    common = set(runs[0][2])
    for _offset, _fingerprint, rows in runs[1:]:
        common.intersection_update(rows)
    if not common:
        raise ValueError("runs contain no complete comparisons in common")

    results = []
    for key in sorted(common):
        draw_rows = []
        for offset, _fingerprint, rows in runs:
            draw_rows.append({"negative_seed_offset": offset, **rows[key]})
        deltas = np.asarray([row["delta"] for row in draw_rows], dtype=np.float64)
        results.append({
            "section": key[0],
            "backbone": key[1],
            "corpus": key[2],
            "a": key[3],
            "b": key[4],
            "view": key[5],
            "n_draws": len(draw_rows),
            "mean_delta": float(deltas.mean()),
            "sd_delta": float(deltas.std(ddof=1)),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "draws": draw_rows,
        })
    return {"negative_seed_offsets": offsets,
            "source_sha256": next(iter(fingerprints)),
            "interpretation": ("Descriptive variation across negative-sampling draws. "
                               "Per-draw family-bootstrap CIs are not pooled; no "
                               "hypothesis tests are performed."),
            "comparisons": results}


def render_markdown(result):
    lines = [
        "# Negative-sampling sensitivity",
        "",
        result["interpretation"],
        "",
        "Offsets: " + ", ".join(map(str, result["negative_seed_offsets"])),
        "",
        "| Backbone | Corpus | Comparison | View | Mean Δ AUC | SD across draws | Range |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in result["comparisons"]:
        lines.append(
            f"| {row['backbone']} | {row['corpus']} | {row['a']} − {row['b']} | "
            f"{row['view']} | {row['mean_delta']:+.4f} | {row['sd_delta']:.4f} | "
            f"[{row['min_delta']:+.4f}, {row['max_delta']:+.4f}] |")
    lines += ["", "Protocol source SHA-256: `" + result["source_sha256"] + "`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", nargs="+",
                        help="protocol result directories, each already summarized")
    parser.add_argument("--output", help="JSON output path")
    args = parser.parse_args()
    try:
        result = compare_runs(args.input_dirs)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = args.output or os.path.join(os.path.abspath(args.input_dirs[0]),
                                         "negative_sampling_sensitivity.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    markdown_path = os.path.splitext(output)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as stream:
        stream.write(render_markdown(result))
    print(f"Wrote {output} and {markdown_path} ({len(result['comparisons'])} common comparisons)")


if __name__ == "__main__":
    main()
