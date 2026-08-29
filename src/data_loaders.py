import os
import pickle
import numpy as np
import scipy.io
import csv
from PIL import Image
import torch


def resolve_image_path(path):
    """Resolves an image path, handling extension casing mismatches on Linux/Colab."""
    if os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for alt_ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        candidate = base + alt_ext
        if os.path.exists(candidate):
            return candidate
    parent = os.path.dirname(path)
    filename = os.path.basename(path).lower()
    if os.path.exists(parent):
        for f in os.listdir(parent):
            if f.lower() == filename:
                return os.path.join(parent, f)
    return None


def load_kinfacew_pairs(root, fold=None):
    """
    Parses KinFaceW-I or KinFaceW-II datasets and returns lists of image paths and labels.
    Preserves official 5-fold cross-validation assignments from meta_data .mat files.

    Args:
        root (str): Absolute path to the KinFaceW-I or KinFaceW-II root folder.
        fold (int, optional): Filter by official fold index (1 to 5). If None, loads all folds.

    Returns:
        pairs (list of tuples): List of (img1_path, img2_path, label, relation_type, fold)
    """
    # Auto-detect nested root directories
    if os.path.exists(root):
        base_name = os.path.basename(root.rstrip("/\\"))
        nested = os.path.join(root, base_name)
        if os.path.exists(nested) and os.path.exists(os.path.join(nested, "meta_data")):
            root = nested

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
            # row[0]: fold (1..5), row[1]: label (1=kin, 0=non-kin), row[2]: img1, row[3]: img2
            row_fold = int(row[0].flat[0])
            if fold is not None and row_fold != fold:
                continue

            label = int(row[1].flat[0])
            img1 = str(row[2].flat[0])
            img2 = str(row[3].flat[0])

            p1 = resolve_image_path(os.path.join(img_dir, img1))
            p2 = resolve_image_path(os.path.join(img_dir, img2))

            if p1 is None or p2 is None:
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
    # We pair random parents and random children from different families to reach 1:1 class balance
    n_kin = len(pairs_list)
    rng = np.random.default_rng(42)
    added = 0
    seen_pairs = set()

    max_attempts = n_kin * 10
    attempts = 0

    while added < n_kin and attempts < max_attempts:
        attempts += 1
        p_idx = rng.integers(0, len(parent_embs_paths))
        c_idx = rng.integers(0, len(child_embs_paths))

        p_path = parent_embs_paths[p_idx]
        c_path = child_embs_paths[c_idx]

        pair_key = (p_path, c_path)
        if pair_key in seen_pairs:
            continue

        # Ensure they are not from the same family
        p_fid = os.path.basename(p_path).split("-")[1]
        c_fid = os.path.basename(c_path).split("-")[1]
        p_folder = os.path.basename(os.path.dirname(p_path))
        c_folder = os.path.basename(os.path.dirname(c_path))

        if p_fid != c_fid or p_folder != c_folder:
            pairs_list.append((p_path, c_path, 0, "ts_non_kin"))
            seen_pairs.add(pair_key)
            added += 1

    return pairs_list


def get_relation_category(rel, img1_path):
    """
    Map relation string to category index for one-hot encoding.
    Categories: 0=father-son, 1=father-daughter, 2=mother-son, 3=mother-daughter/other/non-kin

    Args:
        rel (str): Relation string from dataset
        img1_path (str): Path to first image (used for disambiguation in some cases)

    Returns:
        int: Category index (0-3)
    """
    # Handle KinFaceW & TSKinFace relations - map each to a distinct category (0..3)
    rel_lower = str(rel).lower()

    if "fd" in rel_lower or "fmd_fc" in rel_lower or "father_daughter" in rel_lower:
        return 1  # father-daughter
    elif "fs" in rel_lower or "fms_fc" in rel_lower or "father_son" in rel_lower:
        return 0  # father-son
    elif "md" in rel_lower or "fmd_mc" in rel_lower or "mother_daughter" in rel_lower:
        return 3  # mother-daughter
    elif "ms" in rel_lower or "fms_mc" in rel_lower or "mother_son" in rel_lower:
        return 2  # mother-son

    # Default fallback for any other relation type
    return 3  # catch-all category


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
    if cache_path and os.path.exists(cache_path):
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


def load_fiw_pairs(fiw_root, max_pairs=None):
    """
    Load FIW (Family In the Wild) dataset pairs from `public/FIDs`.

    Parses each family's `mid.csv` to map parent-child relationships (FD, FS, MD, MS).
    Creates a 1:1 balanced set of Kin and Non-Kin pairs across families.
    """
    import glob
    import random

    fids_dir = os.path.join(fiw_root, "FIDs") if os.path.exists(os.path.join(fiw_root, "FIDs")) else fiw_root
    if not os.path.exists(fids_dir):
        print(f"  [WARNING] FIDs directory not found at {fids_dir}")
        return []

    families = sorted([d for d in os.listdir(fids_dir) if os.path.isdir(os.path.join(fids_dir, d))])
    kin_pairs = []
    all_family_images = {}

    for fid in families:
        fpath = os.path.join(fids_dir, fid)
        mid_csv = os.path.join(fpath, "mid.csv")
        if not os.path.exists(mid_csv):
            continue

        with open(mid_csv, "r") as f:
            reader = list(csv.reader(f))
        if len(reader) < 2:
            continue

        header = [x.strip() for x in reader[0]]
        mids = []
        genders = {}
        rel_matrix = {}
        all_family_images[fid] = []

        for row in reader[1:]:
            if not row:
                continue
            mid_id = row[0].strip()
            mids.append(mid_id)
            gender = row[-1].strip().lower()
            genders[mid_id] = gender
            rel_matrix[mid_id] = {}
            for col_idx, m in enumerate(header[1:-2], start=1):
                if col_idx < len(row):
                    try:
                        rel_matrix[mid_id][m] = int(row[col_idx].strip())
                    except ValueError:
                        pass
            m_imgs = glob.glob(os.path.join(fpath, f"MID{mid_id}", "*.jpg")) + glob.glob(os.path.join(fpath, f"MID{mid_id}", "*.png"))
            all_family_images[fid].extend(m_imgs)

        # Generate parent-child kin pairs
        for m1 in mids:
            for m2 in mids:
                if m1 == m2:
                    continue
                code = rel_matrix.get(m1, {}).get(m2, 0)
                if code in (1, 4):  # 1 = child of, 4 = parent of
                    if code == 1:
                        child_id, parent_id = m1, m2
                    else:
                        parent_id, child_id = m1, m2

                    parent_gender = genders.get(parent_id, "m")
                    child_gender = genders.get(child_id, "f")

                    p_imgs = glob.glob(os.path.join(fpath, f"MID{parent_id}", "*.jpg"))
                    c_imgs = glob.glob(os.path.join(fpath, f"MID{child_id}", "*.jpg"))

                    if parent_gender == "m" and child_gender == "f":
                        rel_str = "fd"
                    elif parent_gender == "m" and child_gender == "m":
                        rel_str = "fs"
                    elif parent_gender == "f" and child_gender == "f":
                        rel_str = "md"
                    else:
                        rel_str = "ms"

                    for p_img in p_imgs:
                        for c_img in c_imgs:
                            kin_pairs.append((p_img, c_img, 1, rel_str))

    # Generate equal number of Non-Kin Pairs from different families
    nonkin_pairs = []
    fid_list = [f for f in families if f in all_family_images and all_family_images[f]]
    random.seed(42)

    for p1, p2, _, rel_str in kin_pairs:
        p1_fid = os.path.basename(os.path.dirname(os.path.dirname(p1)))
        other_fids = [f for f in fid_list if f != p1_fid]
        if not other_fids:
            continue
        other_fid = random.choice(other_fids)
        other_img = random.choice(all_family_images[other_fid])
        nonkin_pairs.append((p1, other_img, 0, rel_str))

    total_pairs = kin_pairs + nonkin_pairs
    random.seed(42)
    random.shuffle(total_pairs)

    if max_pairs and max_pairs > 0 and len(total_pairs) > max_pairs:
        half = max_pairs // 2
        sampled_kin = [p for p in total_pairs if p[2] == 1][:half]
        sampled_nonkin = [p for p in total_pairs if p[2] == 0][:half]
        total_pairs = sampled_kin + sampled_nonkin
        random.shuffle(total_pairs)

    print(f"  [FIW] Loaded {len(total_pairs)} pairs (kin={sum(1 for p in total_pairs if p[2]==1)}, non-kin={sum(1 for p in total_pairs if p[2]==0)})")
    return total_pairs


