# -*- coding: utf-8 -*-
"""
MASTER RESEARCH REPORT GENERATOR
Compiles a publication PDF (Quantum_Kinship_Master_Research_Report.pdf) and Markdown summary
aggregating all 12 Priority Module outputs.
NOTE: This version has been corrected to remove inflated claims and hard-coded paths.
"""

import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
outputs_dir = os.path.join(current_dir, "outputs")

def generate_markdown_summary():
    print("\n" + "="*70)
    print("  GENERATING MASTER RESEARCH SUMMARY")
    print("="*70)

    # Generate Markdown Master Summary
    md_summary_path = os.path.join(current_dir, "master_research_summary.md")
    with open(md_summary_path, "w") as f:
        f.write("# Master 12-Module Research Evaluation Summary\n\n")
        f.write("## Important Notes\n\n")
        f.write("> **WARNING**: This research contains known limitations that affect the validity of certain results. ")
        f.write("Please see the limitations section below before interpreting any performance claims.\n\n")
        f.write("## Known Limitations Affecting Results\n\n")
        f.write("1. **FIW Dataset Contamination**: Previous versions contained a bug where 61.2% of FIW 'kin' pairs were actually the same person paired with themselves. ")
        f.write("This has been corrected, but full re-evaluation is needed.\n\n")
        f.write("2. **Fabricated Modules**: Modules 1, 2, 10, and 11 previously contained hard-coded literals presented as experimental results. ")
        f.write("These have been replaced with more honest evaluations, but some components still require rigorous validation.\n\n")
        f.write("3. **Data Leakage**: KinFaceW-I training/testing splits suffered from data leakage, inflating performance metrics.\n\n")
        f.write("4. **FaceNet Preprocessing Mismatch**: Input preprocessing did not match FaceNet's training expectations, likely degrading performance.\n\n")
        f.write("5. **TSKinFace Label Errors**: Half of TSKinFace relation labels were incorrect due to substring matching bugs.\n\n")
        f.write("The results presented below should be interpreted as preliminary and work-in-progress.\n\n")

        f.write("---\n\n")
        f.write("## Module Outputs Location\n\n")
        f.write("All module outputs can be found in the `research_experiments/outputs/` directory:\n\n")
        f.write("- `01_ablation_study/` - Architectural ablation study\n")
        f.write("- `02_sota_literature_comparison/` - SOTA literature comparison\n")
        f.write("- `03_statistical_significance/` - Statistical significance testing\n")
        f.write("- `04_explainability_tsne/` - t-SNE explainability analysis\n")
        f.write("- `05_roc_pr_curves/` - ROC and Precision-Recall curves\n")
        f.write("- `06_threshold_analysis/` - Threshold sensitivity analysis\n")
        f.write("- `07_robustness_degradation/` - Robustness to image degradation\n")
        f.write("- `08_computational_efficiency/` - Computational efficiency analysis\n")
        f.write("- `09_qualitative_error_analysis/` - Qualitative error analysis\n")
        f.write("- `10_cross_dataset/` - Cross-dataset generalization matrix\n")
        f.write("- `11_baseline_comparisons/` - Baseline model comparisons\n")
        f.write("- `12_ensemble_weight_ablation/` - Ensemble fusion weight ablation\n\n")

        f.write("---\n\n")
        f.write("## Preliminary Results (Subject to Validation)\n\n")
        f.write("Based on corrected evaluations where available:\n\n")

        # Try to get actual results from unseen evaluation
        unseen_results_path = os.path.join(project_root, "results", "unseen_metrics", "unseen_evaluation_results.json")
        try:
            if os.path.exists(unseen_results_path):
                with open(unseen_results_path, "r") as f:
                    results = json.load(f)

                f.write("### Unseen Dataset Evaluation (Corrected FIW Loader)\n\n")
                f.write("| Dataset | Pairs | Accuracy | ROC-AUC | F1-Score |\n")
                f.write("|---------|-------|----------|---------|----------|\n")
                for dataset_name, dataset_result in results.items():
                    f.write(f"| {dataset_name} | {dataset_result['n_pairs']} | {dataset_result['accuracy']:.2f}% | {dataset_result['roc_auc']:.4f} | {dataset_result['f1']:.2f}% |\n")
                f.write("\n")
            else:
                f.write("### Unseen Dataset Evaluation\n\n")
                f.write("Results not available. Please run the unseen evaluation script first.\n\n")
        except Exception as e:
            f.write(f"### Unseen Dataset Evaluation\n\n")
            f.write(f"Error loading results: {e}\n\n")

        f.write("### Next Steps for Rigorous Evaluation\n\n")
        f.write("To produce publishable results, the following work is required:\n\n")
        f.write("1. **Re-run all evaluations with corrected pipelines**\n")
        f.write("2. **Implement proper subject/family-disjoint splits** for all datasets\n")
        f.write("3. **Retrain models** with corrected data loading and preprocessing\n")
        f.write("4. **Validate Module 4 (t-SNE)** with actual loaded checkpoints\n")
        f.write("5. **Fix Module 7** to evaluate robustness in image space, not embedding space\n")
        f.write("6. **Use independent baselines** for statistical significance testing (Module 3)\n")
        f.write("7. **Re-extract FaceNet embeddings** with correct preprocessing (160x160, (x-127.5)/128)\n")
        f.write("8. **Implement proper ablation studies** by actually retraining ablated variants\n")
        f.write("9. **Remove all hard-coded paths** to ensure reproducibility\n")
        f.write("10. **Update all documentation** to reflect actual rather than claimed performance\n\n")

        f.write("# Conclusion\n\n")
        f.write("This repository contains novel contributions including:\n")
        f.write("- A relation-conditioned quantum-inspired cross-attention mechanism\n")
        f.write("- A differentiable SWAP test fidelity estimator\n")
        f.write("- A hierarchical meta-ensemble fusion approach\n")
        f.write("- A closed-form analytical fidelity that is 884× faster than quantum circuit simulation\n\n")
        f.write("However, the performance claims in the current manuscript are not supported by rigorous evaluation ")
        f.write("due to the data limitations outlined above. Significant work remains to produce publishable results.\n")

    print(f"[MARKDOWN SUMMARY SAVED] {md_summary_path}")

    # Try to generate a simple PDF report if reportlab is available
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

        print("\n" + "="*70)
        print("  GENERATING PDF RESEARCH SUMMARY")
        print("="*70)

        pdf_filename = "Quantum_Kinship_Research_Summary.pdf"
        pdf_path = os.path.join(current_dir, pdf_filename)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#0F2027'), alignment=1, spaceAfter=12)
        subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=colors.HexColor('#008080'), alignment=1, spaceAfter=8)
        h1_style = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#0F2027'), spaceBefore=10, spaceAfter=4, keepWithNext=True)
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#222222'), spaceAfter=6)

        elements = []

        # Title Banner
        elements.append(Paragraph("QUANTUM-INSPIRED FACIAL KINSHIP VERIFICATION", title_style))
        elements.append(Paragraph("Research Summary & Limitations Disclosure", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#008080"), spaceAfter=12))

        # Limitations Section
        elements.append(Paragraph("IMPORTANT LIMITATIONS & NEXT STEPS", h1_style))
        limitations_text = (
            "The current repository contains known limitations that invalidate certain performance claims. "
            "These include: FIW dataset contamination (same-person pairs), fabricated experimental modules "
            "in research_experiments, data leakage in training/testing splits, FaceNet preprocessing mismatches, "
            "and TSKinFace label errors. Before any publication attempt, these issues must be addressed "
            "through rigorous re-evaluation with corrected pipelines."
        )
        elements.append(Paragraph(limitations_text, body_style))
        elements.append(Spacer(1, 8))

        # Contributions Section
        elements.append(Paragraph("NOVEL CONTRIBUTIONS (PENDING VALIDATION)", h1_style))
        contributions_text = (
            "Despite the evaluation limitations, the following contributions appear promising and warrant "
            "further investigation: (1) Relation-conditioned cross-attention for kinship verification, "
            "(2) Differentiable SWAP test fidelity with quantum-inspired interference, "
            "(3) Hierarchical meta-ensemble for cross-domain generalization, "
            "(4) Closed-form analytical fidelity providing 884× speedup over quantum simulation."
        )
        elements.append(Paragraph(contributions_text, body_style))
        elements.append(Spacer(1, 8))

        # Output Location
        elements.append(Paragraph("OUTPUT FILES LOCATION", h1_style))
        output_text = f"All module outputs are available in: {outputs_dir}"
        elements.append(Paragraph(output_text, body_style))

        doc.build(elements)
        print(f"[PDF SUMMARY GENERATED] {pdf_path}")

    except ImportError:
        print("  [INFO] ReportLab not available, skipping PDF generation")
    except Exception as e:
        print(f"  [WARNING] Could not generate PDF: {e}")

if __name__ == "__main__":
    generate_markdown_summary()