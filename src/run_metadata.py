"""Small provenance helpers for experiment result artifacts."""
import os
import subprocess

from .ts_pairs import family_of_ts


TSKINFACE_NEGATIVE_POLICY = (
    "Split positive pairs by family first; within each side retain one "
    "father-vs-mother negative per available family and sample cross-family "
    "parent-child negatives only among families in that side."
)


def collect_run_metadata(args, datasets=None, project_root=None):
    """Return JSON-serializable CLI, source and TSKinFace protocol metadata."""
    datasets = datasets or {}
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root,
                stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            commit = None

    ts_pairs = datasets.get("TSKinFace", [])
    return {
        "source_commit": commit,
        "cli_args": vars(args).copy() if hasattr(args, "__dict__") else {},
        "tskinface_protocol": {
            "negative_policy": TSKINFACE_NEGATIVE_POLICY,
            "positive_pairs_after_embedding_coverage": len(ts_pairs),
            "families_after_embedding_coverage": len({
                family_of_ts(pair[0]) for pair in ts_pairs
            }),
            # The --cap-per-family option in these runners is a FIW
            # positive-pair cap; it is not applied to TSKinFace.
            "family_cap_applied": False,
            "cap_note": (
                "TSKinFace is not capped by --cap-per-family in these runners; "
                "recorded family count is the post-embedding-coverage count."
            ),
        },
    }
