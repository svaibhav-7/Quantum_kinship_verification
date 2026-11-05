"""Recompute family-cluster intervals from saved formulation predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from rerun_formulations_tskinface import cluster_bootstrap_delta


COMPARISONS = (
    ("formulation_2", "formulation_2_quantum", "formulation_2_control"),
    ("formulation_3", "formulation_3_density", "formulation_3_control"),
    ("formulation_4", "formulation_4_phase", "formulation_4_control"),
    ("formulation_5", "formulation_5_entropy", "formulation_5_control"),
    ("formulation_6", "formulation_6_povm", "formulation_6_control"),
    ("formulation_7", "formulation_7_purity", "formulation_7_control"),
    ("formulation_9", "formulation_9_phase", "formulation_9_control"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="result JSON with saved predictions")
    parser.add_argument(
        "--output",
        type=Path,
        help="new output path; defaults to <input stem>_family_bootstrap.json",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=10042)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_family_bootstrap.json")
    )
    if input_path == output_path:
        parser.error("Output must differ from input; source artifacts are preserved.")
    if output_path.exists():
        parser.error("Output exists; choose a new output path.")

    raw_bytes = input_path.read_bytes()
    artifact = json.loads(raw_bytes.decode("utf-8"))
    results = artifact["results"]
    comparisons = artifact["paired_comparisons"]
    for comparison in comparisons.values():
        if isinstance(comparison, dict):
            for key in ("raw_p", "holm_p", "bh_p"):
                comparison.pop(key, None)
    for index, (name, treatment, control) in enumerate(COMPARISONS):
        comparisons[name]["family_cluster_bootstrap"] = cluster_bootstrap_delta(
            results[treatment]["per_fold"],
            results[control]["per_fold"],
            seed=args.seed + index,
            replicates=args.replicates,
        )

    config = artifact.setdefault("config", {})
    config["bootstrap_analysis"] = {
        "method": (
            "CI-only paired family-cluster bootstrap within each fold; equal-weight "
            "mean of fold-wise AUC differences; no cross-fold score pooling"
        ),
        "replicates": args.replicates,
        "seed": args.seed,
        "source_artifact": input_path.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print("Saved:", output_path)
    for name, _, _ in COMPARISONS:
        bootstrap = comparisons[name]["family_cluster_bootstrap"]
        print(
            "%s: mean fold delta %.6f, family-bootstrap 95%% CI [%.6f, %.6f]"
            % (name, bootstrap["mean_fold_auc_delta"], *bootstrap["ci95"])
        )


if __name__ == "__main__":
    main()
