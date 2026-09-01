# -*- coding: utf-8 -*-
"""Kinship verification CLI.

Single pair (one photo each):
  predict.py --img1 parent.jpg --img2 child.jpg --relation fs --domain fiw

Set-level (several photos per person; +0.077 ROC-AUC where sets exist):
  predict.py --set-a p1.jpg p2.jpg p3.jpg --set-b c1.jpg c2.jpg

Triadic (father + mother + child scored jointly; +0.032 ROC-AUC):
  predict.py --father f.jpg --mother m.jpg --child c.jpg

Batch:
  predict.py --pairs pairs.csv --out results.csv     (img1,img2,relation)
"""
import argparse, csv, os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor import KinshipPredictor, RELATIONS

DEFAULT_CKPT = os.path.join(project_root, "weights", "deploy", "kinship_model.pt")


def main():
    ap = argparse.ArgumentParser(description="Verify kinship between two faces.")
    ap.add_argument("--img1"); ap.add_argument("--img2")
    ap.add_argument("--relation", choices=list(RELATIONS))
    ap.add_argument("--pairs", help="CSV of img1,img2,relation")
    ap.add_argument("--out", help="write CSV results here")
    ap.add_argument("--model", default=DEFAULT_CKPT)
    ap.add_argument("--threshold", type=float, help="override calibrated threshold")
    ap.add_argument("--domain", help="dataset domain for threshold (fiw, kinfacew-i, kinfacew-ii, tskinface)")
    ap.add_argument("--set-a", nargs="+", metavar="IMG",
                    help="all photos of person A (set-level scoring)")
    ap.add_argument("--set-b", nargs="+", metavar="IMG",
                    help="all photos of person B")
    ap.add_argument("--father", help="triadic: father image")
    ap.add_argument("--mother", help="triadic: mother image")
    ap.add_argument("--child", help="triadic: child image")
    args = ap.parse_args()

    # --- triadic -----------------------------------------------------------
    if args.father or args.mother or args.child:
        if not (args.father and args.mother and args.child):
            sys.exit("error: triadic scoring needs --father, --mother and --child")
        from src.set_predictor import TriadPredictor

        path = os.path.join(project_root, "weights", "deploy", "triad_model.pkl")
        if not os.path.exists(path):
            sys.exit("error: triad model missing; run scripts/deploy/package_setlevel.py")
        tp = TriadPredictor(path)
        m = tp.metrics or {}
        print("  triadic model  held-out: %.2f%% acc, %.4f AUC"
              % (m.get("accuracy", 0), m.get("roc_auc", 0)))
        print("  threshold: %.4f\n" % tp.threshold)
        r = tp.predict_images(args.father, args.mother, args.child)
        print("  probability : %.4f" % r["probability"])
        print("  verdict     : %s"
              % ("RELATED (KIN)" if r["is_kin"] else "NOT RELATED"))
        print("  resembles   : %s (alpha=%.2f)" % (r["resembles"], r["alpha"]))
        print("  cos(F,C)=%.4f  cos(M,C)=%.4f" % (r["cos_fc"], r["cos_mc"]))
        return

    # --- set-level ---------------------------------------------------------
    if args.set_a or args.set_b:
        if not (args.set_a and args.set_b):
            sys.exit("error: set-level scoring needs both --set-a and --set-b")
        from src.set_predictor import SetKinshipPredictor

        path = os.path.join(project_root, "weights", "deploy", "set_model.pkl")
        if not os.path.exists(path):
            sys.exit("error: set model missing; run scripts/deploy/package_setlevel.py")
        sp = SetKinshipPredictor(path)
        m = sp.metrics or {}
        print("  set-level model  held-out: %.2f%% acc, %.4f AUC (mean set size %.2f)"
              % (m.get("accuracy", 0), m.get("roc_auc", 0), m.get("mean_set_size", 1)))
        print("  threshold: %.4f\n" % sp.threshold)
        for pth in list(args.set_a) + list(args.set_b):
            if not os.path.exists(pth):
                sys.exit("error: image not found: %s" % pth)
        r = sp.predict_images(args.set_a, args.set_b)
        print("  photos      : %d vs %d" % (r["n_a"], r["n_b"]))
        print("  probability : %.4f" % r["probability"])
        print("  verdict     : %s"
              % ("RELATED (KIN)" if r["is_kin"] else "NOT RELATED"))
        if r["n_a"] == 1 and r["n_b"] == 1:
            print("  note        : one photo each -- this is the single-image path")
        return

    if not os.path.exists(args.model):
        sys.exit(f"error: model not found at {args.model}\n"
                 f"build it with: python scripts/deploy/package_model.py")

    pred = KinshipPredictor(args.model)
    if args.threshold is not None:
        pred.threshold = args.threshold

    m = pred.metrics or {}
    print(f"  model: {os.path.relpath(args.model, project_root)}")
    if m:
        print(f"  held-out: {m.get('accuracy', 0):.2f}% acc, "
              f"{m.get('roc_auc', 0):.4f} AUC, {m.get('fpr', 0):.1f}% FPR")
    print(f"  threshold: {pred.threshold:.4f}  device: {pred.device}\n")

    if args.pairs:
        rows = []
        with open(args.pairs, newline="") as f:
            for r in csv.reader(f):
                if len(r) < 3 or r[0].strip().lower() in ("img1", "image1"):
                    continue
                rows.append((r[0].strip(), r[1].strip(), r[2].strip().lower()))
        if not rows:
            sys.exit("error: no usable rows in CSV (need img1,img2,relation)")

        from src.models_improved import FaceFeatureExtractor
        ext = FaceFeatureExtractor()
        out = []
        for i, (a, b, rel) in enumerate(rows, 1):
            try:
                r = pred.predict_images(a, b, rel, extractor=ext, domain=args.domain)
                out.append((a, b, rel, f"{r['probability']:.4f}", int(r["is_kin"])))
                print(f"  [{i}/{len(rows)}] {r['probability']:.4f}  "
                      f"{'KIN' if r['is_kin'] else 'NOT KIN'}  {os.path.basename(a)} / {os.path.basename(b)}")
            except Exception as e:
                out.append((a, b, rel, "ERROR", str(e)))
                print(f"  [{i}/{len(rows)}] ERROR: {e}")

        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["img1", "img2", "relation", "probability", "is_kin"])
                w.writerows(out)
            print(f"\n  wrote {args.out}")
        return

    if not (args.img1 and args.img2 and args.relation):
        sys.exit("error: give --img1 --img2 --relation, or --pairs FILE.csv")
    for p in (args.img1, args.img2):
        if not os.path.exists(p):
            sys.exit(f"error: image not found: {p}")

    r = pred.predict_images(args.img1, args.img2, args.relation, domain=args.domain)
    verdict = "RELATED (KIN)" if r["is_kin"] else "NOT RELATED"
    print(f"  probability : {r['probability']:.4f}")
    print(f"  verdict     : {verdict}")
    margin = r["probability"] - r["threshold"]
    if abs(margin) < 0.05:
        print(f"  note        : borderline (margin {margin:+.4f}) -- treat as low confidence")


if __name__ == "__main__":
    main()
