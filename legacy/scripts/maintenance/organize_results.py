import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results_dir = ROOT / "results"
plots_dir = results_dir / "plots"
reports_dir = results_dir / "reports"
plots_dir.mkdir(exist_ok=True, parents=True)
reports_dir.mkdir(exist_ok=True, parents=True)

# Move summary plot to plots/
summary_src = results_dir / "model_comparison_summary.png"
if summary_src.exists():
    shutil.move(str(summary_src), str(plots_dir / "model_comparison_summary.png"))

# Consolidate legacy result images into plots/
legacy_plot_sources = [
    results_dir / "evaluation_plots",
    results_dir / "training_metrics",
    results_dir / "fiw_retrained_metrics",
]

for folder in legacy_plot_sources:
    if not folder.exists():
        continue
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            target = plots_dir / path.name
            if target.exists() and target != path:
                target.unlink()
            shutil.move(str(path), str(target))

# Move key JSON summaries to reports/
json_sources = [
    results_dir / "fiw_retrained_metrics" / "fiw_retrained_results.json",
    results_dir / "training_metrics" / "ensemble_live_evaluation.json",
    results_dir / "training_metrics" / "final_evaluation_metrics.json",
    results_dir / "training_metrics" / "fold_results.json",
    results_dir / "training_metrics" / "fold_results_improved.json",
]
for src in json_sources:
    if src.exists():
        shutil.move(str(src), str(reports_dir / src.name))

# Remove empty directories left behind
for folder in [
    results_dir / "evaluation_plots",
    results_dir / "fiw_retrained_metrics",
    results_dir / "training_metrics",
    results_dir / "unseen_metrics",
]:
    if folder.exists() and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()

# Create a small index for humans
readme_path = results_dir / "README.md"
readme_path.write_text(
    "# Results Layout\n\n"
    "- Plots: plots/\n"
    "- Reports: reports/\n"
    "- Raw evaluation data: unseen_metrics/ (if present)\n\n"
    "Use the plots folder for charts and the reports folder for metric summaries.\n",
    encoding="utf-8",
)

print("Results organized successfully")
