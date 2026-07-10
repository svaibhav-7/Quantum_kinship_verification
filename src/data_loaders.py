import os
import pickle
import numpy as np
import scipy.io
from PIL import Image
import torch


def load_kinfacew_pairs(root):
    """
    Parses KinFaceW-I or KinFaceW-II datasets and returns lists of image paths and labels.

    Args:
        root (str): Absolute path to the KinFaceW-I or KinFaceW-II root folder (which contains images/ and meta_data/).

    Returns:
        pairs (list of tuples): List of (img1_path, img2_path, label, relation_type)
    """
    relations = ["fd", "fs", "md", "ms"]
    rel_dirs = {
        "fd": "father-dau",
        "fs": "father-son",
        "md": "mother-dau",
        "ms": "mother-son",
    }

    pairs_list = []
    skipped = 0

    for rel in relations:
        mat_path = os.path.join(root, "meta_data", f"{rel}_pairs.mat")
        if not os.path.exists(mat_path):
            print(f"    [Warning] Mat file not found: {mat_path}")
            continue

        data = scipy.io.loadmat(mat_path)
        pairs = data["pairs"]
        img_dir = os.path.join(root, "images", rel_dirs[rel])

        for row in pairs:
            # mat file format:
            # row[0]: fold, row[1]: label (1=kin, 0=non-kin), row[2]: img1, row[3]: img2
            label = int(row[1].flat[0])
            img1 = str(row[2].flat[0])
            img2 = str(row[3].flat[0])

            p1 = os.path.join(img_dir, img1)
            p2 = os.path.join(img_dir, img2)

            if not os.path.exists(p1) or not os.path.exists(p2):
                skipped += 1
                continue

            pairs_list.append((p1, p2, label, rel))

    if skipped > 0:
        print(f"    [Info] Skipped {skipped} pairs with missing images in {root}")

    return pairs_list


def load_tskinface_pairs(root, max_families=200):
    """
    Parses TSKinFace cropped dataset and returns lists of image paths and labels.

    Args:
        root (str): Absolute path to TSKinFace_cropped folder (contains FMS and FMD).
        max_families (int): Maximum number of families to load.

    Returns:
        pairs (list of tuples): List of (img1_path, img2_path, label, relation_type)
    """
    pairs_list = []
    parent_embs_paths = []
    child_embs_paths = []

    for folder in ["FMS", "FMD"]:
        fdir = os.path.join(root, folder)
        if not os.path.exists(fdir):
            print(f"    [Warning] Folder not found: {fdir}")
            continue

        child_key = "S" if folder == "FMS" else "D"

        # Discover unique family IDs
        families = set()
        for f in os.listdir(fdir):
            if f.endswith(".jpg"):
                parts = f.replace(".jpg", "").split("-")
                if len(parts) == 3:
                    families.add(int(parts[1]))

        families = sorted(families)
        if max_families is not None:
            families = families[:max_families]

        for fid in families:
            f_path = os.path.join(fdir, f"{folder}-{fid}-F.jpg")
            m_path = os.path.join(fdir, f"{folder}-{fid}-M.jpg")
            c_path = os.path.join(fdir, f"{folder}-{fid}-{child_key}.jpg")

            if not all(os.path.exists(p) for p in [f_path, m_path, c_path]):
                continue

            # Create Kin pairs: Father-Child and Mother-Child (label = 1)
            pairs_list.append((f_path, c_path, 1, f"ts_{folder.lower()}_fc"))
            pairs_list.append((m_path, c_path, 1, f"ts_{folder.lower()}_mc"))

            parent_embs_paths.extend([f_path, m_path])
            child_embs_paths.append(c_path)

    # Generate Non-kin pairs: cross-match parents and children (label = 0)
    # We pair random parents and random children from different families to avoid accidental kin
    n_kin = len(pairs_list)
    rng = np.random.default_rng(42)
    added = 0

    # Simple pairing: shuffle parents and children and combine
    shuffled_parents = list(parent_embs_paths)
    shuffled_children = list(child_embs_paths)

    while added < n_kin and len(shuffled_parents) > 0 and len(shuffled_children) > 0:
        p_idx = rng.integers(0, len(shuffled_parents))
        c_idx = rng.integers(0, len(shuffled_children))

        p_path = shuffled_parents[p_idx]
        c_path = shuffled_children[c_idx]

        # Ensure they are not from the same family
        # Path format: .../FMS/FMS-fid-F.jpg and .../FMS/FMS-fid-S.jpg
        p_fid = os.path.basename(p_path).split("-")[1]
        c_fid = os.path.basename(c_path).split("-")[1]
        p_folder = os.path.basename(os.path.dirname(p_path))
        c_folder = os.path.basename(os.path.dirname(c_path))

        if p_fid != c_fid or p_folder != c_folder:
            pairs_list.append((p_path, c_path, 0, "ts_non_kin"))
            added += 1
            # Remove to avoid repeated exact duplicates
            shuffled_parents.pop(p_idx)
            if len(shuffled_children) > c_idx:
                shuffled_children.pop(c_idx)

    return pairs_list


def cache_face_embeddings(pairs, feature_extractor, cache_path):
    """
    Pre-extracts and caches 512-dim face embeddings for all unique images in the pairs.

    Args:
        pairs (list): List of (img1_path, img2_path, label, relation_type) tuples.
        feature_extractor (FaceFeatureExtractor): Pretrained CNN feature extractor.
        cache_path (str): File path to save the cached embeddings.

    Returns:
        cache (dict): Dictionary mapping image path to 512-dim embedding.
    """
    # Find all unique image paths (normalize path case/absolute format for robustness)
    unique_paths = set()
    for p1, p2, _, _ in pairs:
        unique_paths.add(os.path.normcase(os.path.abspath(p1)))
        unique_paths.add(os.path.normcase(os.path.abspath(p2)))
    unique_paths = sorted(list(unique_paths))

    # Load existing cache if available
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                raw_cache = pickle.load(f)
            # Normalize loaded cache keys to handle Windows drive casing mismatches
            cache = {
                os.path.normcase(os.path.abspath(k)): v for k, v in raw_cache.items()
            }
            print(
                f"Loaded existing cache with {len(cache)} embeddings from {cache_path}"
            )
        except Exception as e:
            print(f"Error loading cache file: {e}. Re-extracting embeddings...")
            cache = {}

    # Determine which paths need extraction
    paths_to_extract = [p for p in unique_paths if p not in cache]

    if len(paths_to_extract) > 0:
        print(f"Extracting {len(paths_to_extract)} new face embeddings...")
        # Batch extraction to speed up
        batch_size = 32
        total_batches = (len(paths_to_extract) + batch_size - 1) // batch_size
        for idx, i in enumerate(range(0, len(paths_to_extract), batch_size)):
            # Clean progress reporting without external dependencies
            if (idx + 1) % 5 == 0 or (idx + 1) == total_batches or idx == 0:
                print(
                    f"    [CNN Progress] Batch {idx+1}/{total_batches} ({((idx+1)/total_batches)*100:.1f}%)"
                )
            batch_paths = paths_to_extract[i : i + batch_size]
            try:
                embs = feature_extractor.extract_batch(batch_paths)
                for path, emb in zip(batch_paths, embs):
                    cache[path] = emb
            except Exception as e:
                # Fallback to individual extraction if batch fails
                for path in batch_paths:
                    try:
                        cache[path] = feature_extractor.extract(path)
                    except Exception as ex:
                        print(f"Failed to extract embedding for {path}: {ex}")

        # Save updated cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"Saved updated cache with {len(cache)} embeddings to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache file: {e}")

    return cache


def get_relation_category(rel_str, p_path=None):
    """
    Maps relationship type strings to one of 4 canonical categories:
    0: Father-Daughter (fd)
    1: Father-Son (fs)
    2: Mother-Daughter (md)
    3: Mother-Son (ms)
    """
    rel_str = rel_str.lower()
    if "fd" in rel_str or "father-dau" in rel_str:
        return 0
    elif "fs" in rel_str or "father-son" in rel_str:
        return 1
    elif "md" in rel_str or "mother-dau" in rel_str:
        return 2
    elif "ms" in rel_str or "mother-son" in rel_str:
        return 3
    elif "ts_non_kin" in rel_str and p_path is not None:
        # Determine based on parent type in TSKinFace
        # format: .../FMS/FMS-fid-F.jpg -> Father
        basename = os.path.basename(p_path)
        is_father = "-f" in basename.lower()
        is_fms = "fms" in p_path.lower()
        if is_fms:
            return 1 if is_father else 3  # Father-Son or Mother-Son
        else:
            return 0 if is_father else 2  # Father-Daughter or Mother-Daughter
    else:
        return 0


def prepare_pair_tensors(pairs, cache):
    """
    Converts pairs and their cached embeddings into PyTorch Tensors,
    including one-hot relation representations.

    Args:
        pairs (list): List of (img1_path, img2_path, label, relation_type) tuples.
        cache (dict): Dictionary of cached embeddings.

    Returns:
        emb1_tensor (Tensor): (N, 512) tensor for Person 1.
        emb2_tensor (Tensor): (N, 512) tensor for Person 2.
        labels_tensor (Tensor): (N, 1) tensor of float labels.
        rels_tensor (Tensor): (N, 4) tensor of one-hot relation categories.
    """
    emb1_list, emb2_list, labels_list, rels_list = [], [], [], []

    for p1, p2, label, rel in pairs:
        np1 = os.path.normcase(os.path.abspath(p1))
        np2 = os.path.normcase(os.path.abspath(p2))
        if np1 in cache and np2 in cache:
            emb1_list.append(cache[np1])
            emb2_list.append(cache[np2])
            labels_list.append([float(label)])

            # Map relation to one-hot vector of size 4
            cat = get_relation_category(rel, p1)
            one_hot = [0.0] * 4
            one_hot[cat] = 1.0
            rels_list.append(one_hot)

    emb1_tensor = torch.tensor(np.array(emb1_list), dtype=torch.float32)
    emb2_tensor = torch.tensor(np.array(emb2_list), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32)
    rels_tensor = torch.tensor(np.array(rels_list), dtype=torch.float32)

    return emb1_tensor, emb2_tensor, labels_tensor, rels_tensor
