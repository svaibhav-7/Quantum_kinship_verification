# -*- coding: utf-8 -*-
"""Package the protocol-v2 models into weights/deploy/v2/.

The pair and set models are fitted on the official RFIW train side, calibrated
on its validation side and scored on the available official test pairs. The
pair checkpoint selects among fair, relation-free quantum-off and
relation-free quantum-on models using validation AUC only. The
triadic model uses TSKinFace, the only corpus with triads, and its metrics carry
the colour-only score because that corpus has a strong photo-context cue.
Earlier weights in weights/deploy/ are left in place.
"""
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.fair_protocol import ColourDistance, build_side, split_fit_val
from src.identity_sets import features_to_vector, identity_of_path, pair_set_features
from src.kfold import grouped_kfold
from src.models_fair import FairKinshipModel, RelationFreeHybrid
from src.rfiw import load_identity_pairs, load_labelled_pairs
from src.set_predictor import train_set_model, train_triad_model
from src.splits import family_of
from src.triads import TRIAD_FEATURE_ORDER, find_triads, triad_features
from src.ts_pairs import family_of_ts

OUT = os.path.join(ROOT, "weights", "deploy", "v2")
RFIW = os.path.join(ROOT, "public", "rfiw_track1")
TS_ROOT = os.path.join(ROOT, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CAVEAT = ("Official RFIW split; test uses only the 74% of pairs whose faces are "
          "available. Threshold calibrated on the official validation side.")


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def load_cache(name):
    with open(os.path.join(ROOT, "weights", "caches", name), "rb") as stream:
        return {norm(k): np.asarray(v, dtype=np.float32) for k, v in pickle.load(stream).items()}


def covered(pairs, cache):
    return [p for p in pairs if norm(p[0]) in cache and norm(p[1]) in cache]


def as_tensors(pairs, cache):
    a = torch.as_tensor(np.stack([cache[norm(p[0])] for p in pairs]), device=DEVICE)
    b = torch.as_tensor(np.stack([cache[norm(p[1])] for p in pairs]), device=DEVICE)
    y = torch.as_tensor(np.array([float(p[2]) for p in pairs], dtype=np.float32), device=DEVICE)
    return a, b, y


@torch.no_grad()
def scores(model, data, model_type):
    model.eval()
    values = []
    for i in range(0, len(data[2]), 4096):
        a, b = data[0][i:i + 4096], data[1][i:i + 4096]
        if model_type == "fair":
            logits = model.forward_logits(a, b)
        else:
            relations = torch.zeros((len(a), 4), dtype=a.dtype, device=a.device)
            logits = model.forward_logits(a, b, relations)
        values.append(logits.sigmoid().view(-1).cpu())
    return torch.cat(values).numpy()


def train_pair_model(fit, val):
    """Choose the model family and seed with highest validation AUC."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    val_y = val[2].cpu().numpy()
    candidates = (("fair", None, 3e-3), ("relation_free_hybrid", False, 1e-3),
                  ("relation_free_hybrid", True, 1e-3))
    best_auc, best_state, best_type, best_seed, best_quantum = -1.0, None, None, None, False
    for model_type, use_quantum, learning_rate in candidates:
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            if model_type == "fair":
                model = FairKinshipModel().to(DEVICE)
            else:
                model = RelationFreeHybrid(use_quantum=use_quantum).to(DEVICE)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-2)
            since = 0
            fit_relations = torch.zeros((len(fit[2]), 4), dtype=fit[0].dtype, device=DEVICE)
            for _ in range(40):
                model.train()
                order = torch.randperm(len(fit[2]), device=DEVICE)
                for start in range(0, len(order), 512):
                    index = order[start:start + 512]
                    optimizer.zero_grad()
                    if model_type == "fair":
                        logits = model.forward_logits(fit[0][index], fit[1][index])
                    else:
                        logits = model.forward_logits(
                            fit[0][index], fit[1][index], fit_relations[index])
                    loss_fn(logits.view(-1), fit[2][index]).backward()
                    optimizer.step()
                value = roc_auc_score(val_y, scores(model, val, model_type))
                if value > best_auc:
                    best_auc, since = value, 0
                    best_type, best_seed, best_quantum = model_type, seed, bool(use_quantum)
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                else:
                    since += 1
                    if since >= 8:
                        break
    if best_type is None or best_state is None:
        raise RuntimeError("No deployment candidate was trained")
    model = (FairKinshipModel() if best_type == "fair"
             else RelationFreeHybrid(use_quantum=best_quantum)).to(DEVICE)
    model.load_state_dict(best_state)
    return model, float(best_auc), best_type, best_seed


def rfiw_sides(cache):
    fit = covered(load_identity_pairs(os.path.join(RFIW, "train-pairs.csv"),
                                      os.path.join(RFIW, "train-faces"),
                                      cap_per_identity_pair=30, seed=SEED), cache)
    val = covered(load_identity_pairs(os.path.join(RFIW, "val-pairs.csv"),
                                      os.path.join(RFIW, "val-faces"),
                                      cap_per_identity_pair=30, seed=SEED), cache)
    test = covered(load_labelled_pairs(os.path.join(RFIW, "test-pairs.csv"),
                                       os.path.join(RFIW, "test-fids")), cache)
    return fit, val, test


def package_pair(cache, sides):
    fit_pos, val_pos, test = sides
    fit = as_tensors(build_side(fit_pos, family_of, SEED), cache)
    val = as_tensors(build_side(val_pos, family_of, SEED + 1), cache)
    held = as_tensors(test, cache)
    model, val_auc, model_type, selected_seed = train_pair_model(fit, val)
    threshold = calibrate_threshold(val[2].cpu().numpy(), scores(model, val, model_type), "accuracy",
                                    max_fpr=0.25, smooth=True)
    test_scores = scores(model, held, model_type)
    y = held[2].cpu().numpy()
    point = operating_point(y, test_scores, threshold)
    metrics = {"accuracy": point["accuracy"], "roc_auc": float(roc_auc_score(y, test_scores)),
               "fpr": point["fpr"], "val_auc": val_auc, "selected_model": model_type,
               "selected_seed": selected_seed, "n_test_pairs": len(y), "evaluation": CAVEAT}
    torch.save({"model_type": model_type,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "threshold": float(threshold),
                "use_quantum": bool(getattr(model, "use_quantum", False)), "metrics": metrics,
                "preprocessing": "uncropped"}, os.path.join(OUT, "kinship_model.pt"))
    return metrics


def identity_side(positives, seed):
    """Matched negatives at identity level; each identity pair appears once."""
    rng = random.Random(seed)
    unique = {}
    for a, b, _l, relation in positives:
        unique[(identity_of_path(a), identity_of_path(b))] = relation
    donors = defaultdict(list)
    for (_a, b), relation in unique.items():
        donors[relation].append(b)
    out = []
    for (a, b), relation in unique.items():
        out.append((a, b, 1))
        for _ in range(64):
            other = donors[relation][rng.randrange(len(donors[relation]))]
            if other.split("/")[0] != a.split("/")[0]:
                out.append((a, other, 0))
                break
    return out


def galleries(pairs, cache):
    out = defaultdict(set)
    for a, b, _l, _r in pairs:
        out[identity_of_path(a)].add(a)
        out[identity_of_path(b)].add(b)
    return {k: np.stack([cache[norm(p)] for p in sorted(v)]) for k, v in out.items()}


def full_sets(csv_name, faces, cache):
    return covered(load_identity_pairs(os.path.join(RFIW, csv_name),
                                       os.path.join(RFIW, faces), seed=SEED), cache)


def package_set(cache, sides):
    fit_pos = full_sets("train-pairs.csv", "train-faces", cache)
    val_pos = full_sets("val-pairs.csv", "val-faces", cache)
    test = sides[2]
    path = os.path.join(OUT, "set_model.pkl")
    train_set_model(galleries(fit_pos, cache), identity_side(fit_pos, SEED), path, seed=SEED,
                    val_galleries=galleries(val_pos, cache),
                    val_pairs=identity_side(val_pos, SEED + 1))
    with open(path, "rb") as stream:
        stored = pickle.load(stream)

    test_galleries = galleries(test, cache)
    labels = {}
    for a, b, label, _r in test:
        labels[(identity_of_path(a), identity_of_path(b))] = label
    X, y = [], []
    for (a, b), label in labels.items():
        X.append(features_to_vector(pair_set_features(test_galleries[a], test_galleries[b])))
        y.append(label)
    probabilities = stored["clf"].predict_proba(stored["scaler"].transform(np.stack(X)))[:, 1]
    y = np.array(y)
    point = operating_point(y, probabilities, stored["threshold"])
    stored["metrics"] = {"accuracy": point["accuracy"],
                         "roc_auc": float(roc_auc_score(y, probabilities)),
                         "fpr": point["fpr"], "n_test_identity_pairs": int(len(y)),
                         "evaluation": CAVEAT}
    with open(path, "wb") as stream:
        pickle.dump(stored, stream)
    return stored["metrics"]


def package_triad(cache):
    colour = ColourDistance()
    triads = {family_of_ts(t["F"]): t for t in find_triads(TS_ROOT).values()
              if all(norm(p) in cache for p in t.values())}
    triads.pop("FMD-216", None)  # stored twice; duplicates FMD-205
    embed = lambda path: cache[norm(path)]
    keys = sorted(triads)
    random.Random(SEED).shuffle(keys)
    n = len(keys) // 5
    test_f, val_f, fit_f = keys[:n], keys[n:2 * n], keys[2 * n:]

    def rows(families, seed):
        rng = random.Random(seed)
        role = lambda f: os.path.basename(triads[f]["C"]).rsplit(".", 1)[0].split("-")[-1]
        out = []
        for family in families:
            t = triads[family]
            out.append((embed(t["F"]), embed(t["M"]), embed(t["C"]), 1))
            donors = [f for f in families if f != family and role(f) == role(family)]
            out.append((embed(t["F"]), embed(t["M"]), embed(triads[rng.choice(donors)]["C"]), 0))
        return out

    def colour_only(families, seed):
        rng = random.Random(seed)
        role = lambda f: os.path.basename(triads[f]["C"]).rsplit(".", 1)[0].split("-")[-1]
        y, values = [], []
        for family in families:
            t = triads[family]
            donors = [f for f in families if f != family and role(f) == role(family)]
            for child, label in ((t["C"], 1), (triads[rng.choice(donors)]["C"], 0)):
                y.append(label)
                values.append(-min(colour(t["F"], child), colour(t["M"], child)))
        return float(roc_auc_score(y, values))

    path = os.path.join(OUT, "triad_model.pkl")
    train_triad_model(rows(fit_f, SEED), path, seed=SEED, val_rows=rows(val_f, SEED + 1))
    with open(path, "rb") as stream:
        stored = pickle.load(stream)
    test_rows = rows(test_f, SEED + 2)
    X = np.array([[triad_features(F, M, C)[k] for k in TRIAD_FEATURE_ORDER]
                  for F, M, C, _l in test_rows])
    y = np.array([row[3] for row in test_rows])
    probabilities = stored["clf"].predict_proba(stored["scaler"].transform(X))[:, 1]
    point = operating_point(y, probabilities, stored["threshold"])
    stored["metrics"] = {
        "accuracy": point["accuracy"], "roc_auc": float(roc_auc_score(y, probabilities)),
        "fpr": point["fpr"], "n_test_rows": int(len(y)),
        "colour_only_roc_auc": colour_only(test_f, SEED + 2),
        "evaluation": ("TSKinFace family-disjoint holdout. Colour alone also separates this "
                       "corpus, so these figures include photo context and overstate kinship.")}
    with open(path, "wb") as stream:
        pickle.dump(stored, stream)
    return stored["metrics"]


def main():
    global OUT
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUT,
                        help="deployment artifact directory; use a new path to stage a candidate")
    args = parser.parse_args()
    OUT = os.path.abspath(args.output_dir)
    os.makedirs(OUT, exist_ok=True)
    cache = load_cache("rfiw_track1_cache.pkl")
    sides = rfiw_sides(cache)
    print("  fit %d, val %d, test %d" % tuple(len(s) for s in sides), flush=True)
    result = {"pair": package_pair(cache, sides)}
    print("  pair", result["pair"], flush=True)
    result["set"] = package_set(cache, sides)
    print("  set", result["set"], flush=True)
    result["triad"] = package_triad(load_cache("all_datasets_cache.pkl"))
    print("  triad", result["triad"], flush=True)
    target = os.path.join(OUT, "deploy_metrics.json")
    with open(target, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print("  saved", OUT, flush=True)
    print("  metrics", target, flush=True)


if __name__ == "__main__":
    main()
