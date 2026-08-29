# -*- coding: utf-8 -*-
"""
COMMON DATA UTILITY for Master Model Comparison (4 Models x 12 Modules)

Loads the 4 kinship datasets (KinFaceW-I, KinFaceW-II, TSKinFace, FIW)
and builds basename-based embedding lookups from the cached embeddings pickles.
This decouples the cached embeddings (which were built on a different machine
with different absolute paths) from the local dataset paths.

Usage:
    from common_data import get_datasets, get_predictions
"""

import os
import sys
import pickle
import numpy as np
import torch

# ------------------------------------------------------------------
# Project root resolution
# ------------------------------------------------------------------
_current = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(_current)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loaders import (
    load_kinfacew_pairs,
    load_tskinface_pairs,
    prepare_pair_tensors,
    get_relation_category,
)
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

# ------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------

def _load_emb_cache(path):
    """Load a pickle embedding cache and return a {lowercase-basename: embedding} dict."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {os.path.basename(k).lower(): v for k, v in raw.items()}


def get_datasets():
    """
    Returns dict: {dataset_name: (emb1, emb2, labels, rels, pair_info)}
    where emb1/emb2/rels are Tensors, labels is a Tensor (N,1),
    and pair_info is a list of (p1_path, p2_path, label, rel_str).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Main cache (KinFaceW-I, KinFaceW-II, TSKinFace)
    main_cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
    main_cache = _load_emb_cache(main_cache_path)

    # FIW cache
    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    fiw_cache = _load_emb_cache(fiw_cache_path)

    datasets = {}

    # ---- KinFaceW-I ----
    k1_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I"))
    k1_t = prepare_pair_tensors_basename(k1_pairs, main_cache)
    datasets["KinFaceW-I"] = (k1_t[0], k1_t[1], k1_t[2], k1_t[3], k1_pairs)

    # ---- KinFaceW-II ----
    k2_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-II"))
    k2_t = prepare_pair_tensors_basename(k2_pairs, main_cache)
    datasets["KinFaceW-II"] = (k2_t[0], k2_t[1], k2_t[2], k2_t[3], k2_pairs)

# ---- TSKinFace ----
    ts_pairs = load_tskinface_pairs(
        os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
    )
    ts_t = prepare_pair_tensors_basename(ts_pairs, main_cache, extract_missing=False)
    if len(ts_t[2]) > 0:
        datasets["TSKinFace"] = (ts_t[0], ts_t[1], ts_t[2], ts_t[3], ts_pairs)
    else:
        print("  [SKIP] TSKinFace has no cached embeddings; skipping.")

    # ---- FIW ----
    fiw_pairs = load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500)
    fiw_t = prepare_pair_tensors_basename(fiw_pairs, fiw_cache)
    datasets["FIW"] = (fiw_t[0], fiw_t[1], fiw_t[2], fiw_t[3], fiw_pairs)

    return datasets


def prepare_pair_tensors_basename(pairs, cache_basename, extract_missing=False):
    """
    Convert pairs to tensors using a lowercase-basename-keyed embedding cache.
    Optionally re-extracts missing embeddings (via FaceFeatureExtractor).
    """
    emb1_list, emb2_list, labels_list, rels_list = [], [], [], []
    missing = []

    for p1, p2, label, rel in pairs:
        b1 = os.path.basename(p1).lower()
        b2 = os.path.basename(p2).lower()
        if b1 in cache_basename and b2 in cache_basename:
            emb1_list.append(cache_basename[b1])
            emb2_list.append(cache_basename[b2])
            labels_list.append([float(label)])
            cat = get_relation_category(rel, p1)
            one_hot = [0.0] * 4
            one_hot[cat] = 1.0
            rels_list.append(one_hot)
        else:
            missing.append((p1, b1, p2, b2))

    if missing and extract_missing:
        print(f"  [WARN] {len(missing)} pairs missing cached embeddings; attempting extraction...")
        try:
            from src.models_improved import FaceFeatureExtractor
            extractor = FaceFeatureExtractor()
            for p1, b1, p2, b2 in missing:
                try:
                    e1 = extractor.extract(p1)
                    e2 = extractor.extract(p2)
                    cache_basename[b1] = e1
                    cache_basename[b2] = e2
                    emb1_list.append(e1)
                    emb2_list.append(e2)
                    # label/rel need pairing info; reconstruct from pairs
                except Exception as ex:
                    print(f"        [SKIP] {b1} / {b2}: {ex}")
        except Exception as ex:
            print(f"  [WARN] Could not extract missing embeddings: {ex}")

    if not emb1_list:
        return (torch.zeros(0, 512), torch.zeros(0, 512),
                torch.zeros(0, 1), torch.zeros(0, 4))

    emb1_tensor = torch.tensor(np.array(emb1_list), dtype=torch.float32)
    emb2_tensor = torch.tensor(np.array(emb2_list), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32)
    rels_tensor = torch.tensor(np.array(rels_list), dtype=torch.float32)

    return emb1_tensor, emb2_tensor, labels_tensor, rels_tensor


def get_predictions(model, emb1, emb2, rels, batch_size=128):
    """
    Return per-dataset prediction scores (numpy array) by batching model forward.
    model must be loaded and moved to device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n = len(emb1)
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_e1 = emb1[i:i + batch_size].to(device)
            b_e2 = emb2[i:i + batch_size].to(device)
            b_rel = rels[i:i + batch_size].to(device)
            p = model(b_e1, b_e2, b_rel)
            preds.append(p.cpu().view(-1))
    if not preds:
        return np.array([])
    return torch.cat(preds, dim=0).numpy()


def get_optimal_threshold(model_key):
    """Return the recorded optimal threshold for a model key."""
    thresholds = {
        "ensemble_kinship_full": 0.5279678702354431,
        "meta_ensemble_kinship": 0.5216,
        "ensemble_kinship_fiw": 0.5279678702354431,
        "best_checkpoint": 0.5,
    }
    return thresholds.get(model_key, 0.5)


if __name__ == "__main__":
    dsets = get_datasets()
    for name, (e1, e2, y, r, pairs) in dsets.items():
        print(f"{name}: {len(y)} pairs, emb {tuple(e1.shape)}, rel {tuple(r.shape)}")
