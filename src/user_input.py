"""Turning arbitrary user photos into embeddings the models can score.

User photos differ from benchmark images in two ways that matter:

1. They are not pre-cropped. Dataset images are already tight face crops; a
   holiday photo is not. We therefore run MTCNN face detection by default.
   Measured on already-cropped dataset faces, detection changes the embedding
   (cosine 0.944 against the uncropped version) but *not* the decision
   (ROC-AUC 0.786 against 0.791 on the same pairs), so enabling it by default
   is safe and is what uncropped input requires.

2. They arrive as paths, URLs or whole folders rather than a curated list.

Face detection runs on CPU: the torchvision NMS operator MTCNN relies on is
not available in this CUDA build. Detection is cheap relative to embedding.
"""

import os

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class InputError(Exception):
    """A user-facing problem with the supplied photos."""


def is_url(s):
    return str(s).lower().startswith(("http://", "https://"))


def _download(url, dest_dir):
    import hashlib
    import urllib.parse

    import requests

    name = os.path.basename(urllib.parse.urlparse(url).path) or "image"
    if not name.lower().endswith(IMAGE_EXTS):
        name += ".jpg"
    # Prefix with a hash so two URLs sharing a basename cannot collide.
    stem = hashlib.sha1(url.encode()).hexdigest()[:8]
    path = os.path.join(dest_dir, f"{stem}_{name}")

    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "kinship-verify/1.0"})
        r.raise_for_status()
    except Exception as e:
        raise InputError(f"could not download {url}: {e}")

    ctype = r.headers.get("content-type", "")
    if ctype and not ctype.startswith("image/"):
        raise InputError(f"{url} returned {ctype or 'unknown type'}, not an image")

    with open(path, "wb") as f:
        f.write(r.content)
    return path


def resolve_inputs(items, download_dir=None):
    """Expand paths, directories and URLs into a de-duplicated list of files."""
    out, seen = [], set()
    for item in items:
        item = str(item).strip().strip('"').strip("'")
        if not item:
            continue

        if is_url(item):
            if download_dir is None:
                import tempfile

                download_dir = tempfile.mkdtemp(prefix="kinship_dl_")
            paths = [_download(item, download_dir)]
        elif os.path.isdir(item):
            paths = sorted(
                os.path.join(item, f) for f in os.listdir(item)
                if f.lower().endswith(IMAGE_EXTS))
            if not paths:
                raise InputError(f"no images found in folder: {item}")
        elif os.path.isfile(item):
            paths = [item]
        else:
            raise InputError(f"not found: {item}")

        for p in paths:
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                out.append(p)

    if not out:
        raise InputError("no images supplied")
    return out


_EXTRACTOR = {"net": None, "mtcnn": None, "plain": None}


def _init_models():
    if _EXTRACTOR["net"] is not None:
        return
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from torchvision import transforms

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _EXTRACTOR["dev"] = dev
    _EXTRACTOR["net"] = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
    # CPU: the NMS op MTCNN needs is missing from this CUDA build.
    _EXTRACTOR["mtcnn"] = MTCNN(image_size=160, margin=20, post_process=True,
                                select_largest=True, device=torch.device("cpu"))
    _EXTRACTOR["plain"] = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [128 / 255] * 3),
    ])


def extract_faces(paths, detect=True):
    """Embed each photo. Returns (embeddings, failures).

    failures is [(path, reason)] so the caller can tell the user which photo
    was unusable and why, rather than silently scoring fewer images.
    """
    import torch
    from PIL import Image

    _init_models()
    dev = _EXTRACTOR["dev"]
    embs, failed = [], []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            failed.append((p, f"could not open image ({e})"))
            continue

        try:
            with torch.no_grad():
                if detect:
                    face = _EXTRACTOR["mtcnn"](img)
                    if face is None:
                        failed.append((p, "no face detected"))
                        continue
                    v = _EXTRACTOR["net"](face.unsqueeze(0).to(dev))[0]
                else:
                    t = _EXTRACTOR["plain"](img).unsqueeze(0).to(dev)
                    v = _EXTRACTOR["net"](t)[0]
            v = v.cpu().numpy()
            embs.append(v / (np.linalg.norm(v) + 1e-12))
        except Exception as e:
            failed.append((p, f"embedding failed ({e})"))

    return embs, failed


def confidence_label(prob, threshold, near=0.08, wide=0.25):
    """How far the score sits from the decision boundary.

    Held-out accuracy is roughly 73-77%, so about one call in four is wrong.
    A score sitting on the threshold deserves to be reported as uncertain
    rather than as a verdict.
    """
    d = abs(float(prob) - float(threshold))
    if d < near:
        return "borderline"
    if d < wide:
        return "moderate"
    return "high"
