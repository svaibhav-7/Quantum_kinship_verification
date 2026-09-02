#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kinship verification for your own photos.

Run with no arguments for an interactive prompt:

    python verify_kinship.py

Or pass photos directly. Each person may be given as one photo, several
photos, a folder, or a URL:

    python verify_kinship.py --person-a mum.jpg --person-b kid.jpg
    python verify_kinship.py --person-a ./mum_photos/ --person-b kid1.jpg kid2.jpg
    python verify_kinship.py --person-a https://site/a.jpg --person-b b.jpg

Triadic mode, when both parents are available (more accurate):

    python verify_kinship.py --father f.jpg --mother m.jpg --child c.jpg

Supplying several photos of a person is worth about +0.077 ROC-AUC on the
benchmark that supports it, so use every photo you have.
"""
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.user_input import (InputError, confidence_label, extract_faces,
                            resolve_inputs)

SET_MODEL = os.path.join(PROJECT_ROOT, "weights", "deploy", "set_model.pkl")
TRIAD_MODEL = os.path.join(PROJECT_ROOT, "weights", "deploy", "triad_model.pkl")

BAR = "=" * 62


def detect_default_for(predictor):
    """Face detection must match how the model was fitted.

    The embeddings a model was fitted on and the embeddings it is served must
    be produced the same way. Measured on 387 person-pairs, serving
    MTCNN-cropped embeddings to a model fitted on uncropped ones flipped
    30.2% of verdicts (mean score shift 0.20) even though aggregate ROC-AUC
    looked unchanged -- which is why aggregate AUC is the wrong check.
    """
    return getattr(predictor, "preprocessing", "uncropped") == "cropped"


def _rule(title=""):
    print(BAR if not title else f"\n{title}\n" + "-" * 62)


def _gather(label, raw, detect):
    """Resolve inputs, embed, and report anything unusable."""
    paths = resolve_inputs(raw)
    embs, failed = extract_faces(paths, detect=detect)

    print(f"  {label}: {len(embs)} usable photo(s) of {len(paths)} supplied")
    for p, why in failed:
        print(f"    skipped {os.path.basename(p)} -- {why}")

    if not embs:
        raise InputError(
            f"no usable face found for {label}. "
            "Try a clearer, front-facing photo, or pass --no-detect if the "
            "images are already tight face crops.")
    return np.stack(embs)


def _verdict(prob, threshold, metrics):
    conf = confidence_label(prob, threshold)
    kin = prob >= threshold

    _rule("RESULT")
    print(f"  score      : {prob:.4f}   (decides KIN at >= {threshold:.4f})")
    print(f"  verdict    : {'RELATED (KIN)' if kin else 'NOT RELATED'}")
    print(f"  confidence : {conf}")

    if conf == "borderline":
        print("\n  This score sits close to the decision boundary. Treat it as")
        print("  inconclusive rather than as an answer.")

    acc = metrics.get("accuracy")
    if acc:
        print(f"\n  For context: this model is right about {acc:.0f}% of the time")
        print("  on held-out families it never saw in training. It is a research")
        print("  prototype -- do not use it for any decision that affects people.")


def run_pair(a_raw, b_raw, detect):
    from src.set_predictor import SetKinshipPredictor

    if not os.path.exists(SET_MODEL):
        sys.exit("model missing. Run: python scripts/deploy/package_setlevel.py")

    pred = SetKinshipPredictor(SET_MODEL)
    if detect is None:
        detect = detect_default_for(pred)
    _rule("PHOTOS")
    print(f"  model was fitted on {pred.preprocessing} faces; "
          f"detection {'ON' if detect else 'OFF'}")
    A = _gather("person A", a_raw, detect)
    B = _gather("person B", b_raw, detect)

    r = pred.predict_sets(A, B)
    _verdict(r["probability"], pred.threshold, pred.metrics)

    if r["n_a"] == 1 and r["n_b"] == 1:
        print("\n  Tip: one photo each was used. Several photos per person is")
        print("  measurably more reliable -- pass a folder or multiple files.")


def run_triad(f_raw, m_raw, c_raw, detect):
    from src.set_predictor import TriadPredictor

    if not os.path.exists(TRIAD_MODEL):
        sys.exit("model missing. Run: python scripts/deploy/package_setlevel.py")

    pred = TriadPredictor(TRIAD_MODEL)
    if detect is None:
        detect = detect_default_for(pred)
    _rule("PHOTOS")
    print(f"  model was fitted on {pred.preprocessing} faces; "
          f"detection {'ON' if detect else 'OFF'}")
    F = _gather("father", f_raw, detect).mean(axis=0)
    M = _gather("mother", m_raw, detect).mean(axis=0)
    C = _gather("child", c_raw, detect).mean(axis=0)

    r = pred.predict_triad(F, M, C)
    _verdict(r["probability"], pred.threshold, pred.metrics)
    print(f"\n  resembles  : {r['resembles']} more (alpha = {r['alpha']:.2f})")
    print(f"  similarity : father {r['cos_fc']:.3f}   mother {r['cos_mc']:.3f}")


def _ask(prompt):
    """Read a whitespace-separated list of photos from the user."""
    raw = input(prompt).strip()
    if not raw:
        return []
    # Keep quoted paths with spaces intact.
    import shlex

    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def interactive(detect):
    print(BAR)
    print("  KINSHIP VERIFICATION")
    print(BAR)
    print("\n  Give photos of two people and I will estimate whether they are")
    print("  biologically related. For each person you may enter:")
    print("    - one photo        mum.jpg")
    print("    - several photos   a.jpg b.jpg c.jpg   (more is better)")
    print("    - a folder         ./mum_photos/")
    print("    - a URL            https://site/photo.jpg")
    print("\n  Enter 't' for triadic mode (father + mother + child).")

    first = _ask("\n  Person A photos (or 't'): ")
    if len(first) == 1 and first[0].lower() == "t":
        f = _ask("  Father photo(s): ")
        m = _ask("  Mother photo(s): ")
        c = _ask("  Child photo(s) : ")
        if not (f and m and c):
            sys.exit("  all three are required for triadic mode")
        run_triad(f, m, c, detect)
        return

    b = _ask("  Person B photos: ")
    if not (first and b):
        sys.exit("  photos are required for both people")
    run_pair(first, b, detect)


def main():
    ap = argparse.ArgumentParser(
        description="Estimate whether two people are biologically related.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--person-a", nargs="+", metavar="PHOTO",
                    help="photo(s), folder or URL for the first person")
    ap.add_argument("--person-b", nargs="+", metavar="PHOTO",
                    help="photo(s), folder or URL for the second person")
    ap.add_argument("--father", nargs="+", metavar="PHOTO")
    ap.add_argument("--mother", nargs="+", metavar="PHOTO")
    ap.add_argument("--child", nargs="+", metavar="PHOTO")
    ap.add_argument("--detect", dest="detect", action="store_true",
                    default=None,
                    help="force face detection on")
    ap.add_argument("--no-detect", dest="detect", action="store_false",
                    help="force face detection off")
    args = ap.parse_args()
    # Left unset, the tool follows whatever the model was fitted on.
    detect = args.detect

    try:
        if args.father or args.mother or args.child:
            if not (args.father and args.mother and args.child):
                sys.exit("triadic mode needs --father, --mother and --child")
            run_triad(args.father, args.mother, args.child, detect)
        elif args.person_a or args.person_b:
            if not (args.person_a and args.person_b):
                sys.exit("both --person-a and --person-b are required")
            run_pair(args.person_a, args.person_b, detect)
        else:
            interactive(detect)
    except InputError as e:
        sys.exit(f"\n  error: {e}")
    except KeyboardInterrupt:
        sys.exit("\n  cancelled")


if __name__ == "__main__":
    main()
