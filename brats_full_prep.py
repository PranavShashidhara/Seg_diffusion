#!/usr/bin/env python3
"""
BraTS 2021 full data prep (LOCAL):
- Recursively extract archives (.tar/.tar.gz/.tgz/.zip) with path-traversal protection
- Normalize folder layout into BraTS2021_XXXXX case directories (if flat)
- Export per-patient 2D PNGs for FLAIR, T1, T1CE, T2 + mask (class indices) + colored preview
- Patient-level train/val/test split
- Write manifest.csv

Usage (examples):
  python brats_full_prep.py \
    --src "/data/brats_raw" \
    --extract-dir "/data/brats_extracted" \
    --out-root "/data/BraTS2021_2D_perpatient" \
    --mask-mode 3class \
    --num-workers 8

  # Tumor-only slices +/− 1 neighbor
  python brats_full_prep.py \
    --src "/data/brats_raw" \
    --extract-dir "/data/brats_extracted" \
    --out-root "/data/BraTS2021_2D_perpatient" \
    --mask-mode 3class \
    --exclude-empty-masks --add-context 1 \
    --num-workers 8
"""

import argparse, os, re, glob, random, logging, shutil, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import nibabel as nib
import pandas as pd
from skimage.io import imsave
from tqdm import tqdm

# --------------------------- Logging ---------------------------

def setup_logging(verbosity: int):
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

# ----------------------- Archive Handling ----------------------

ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".zip")

def _safe_extract_tar(tf: tarfile.TarFile, dest: str):
    dest = os.path.realpath(dest)
    for m in tf.getmembers():
        target = os.path.realpath(os.path.join(dest, m.name))
        if not target.startswith(dest):
            raise Exception("Blocked path traversal in tar")
    tf.extractall(dest)

def _extract_one(archive_path: str, dest_dir: str) -> bool:
    ap = archive_path.lower()
    try:
        if ap.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, mode="r:*") as tf:
                _safe_extract_tar(tf, dest_dir)
            return True
        if ap.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
            return True
    except Exception as e:
        logging.warning("Failed extracting %s: %s", archive_path, e)
    return False

def recursive_extract(src_dir: str, extract_dir: str, passes: int = 6):
    os.makedirs(extract_dir, exist_ok=True)

    # Pass 0: extract top-level archives found in src_dir into extract_dir
    top_archives = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.lower().endswith(ARCHIVE_EXTS)
    ]
    if top_archives:
        logging.info("Top-level archives: %d", len(top_archives))
    for a in top_archives:
        logging.info("Extracting: %s", a)
        _extract_one(a, extract_dir)

    # Passes 1..N: keep extracting nested archives found inside extract_dir
    for p in range(1, passes + 1):
        nested = []
        for ext in ("*.tar", "*.tar.gz", "*.tgz", "*.zip"):
            nested += glob.glob(os.path.join(extract_dir, "**", ext), recursive=True)
        if not nested:
            logging.info("No more nested archives after pass %d", p - 1)
            break
        logging.info("Pass %d: %d nested archives", p, len(nested))
        for n in nested:
            d = os.path.dirname(n)
            if _extract_one(n, d):
                try:
                    os.remove(n)
                except Exception:
                    pass

# ----------------------- Case Normalization --------------------

CASE_PAT = re.compile(r"(BraTS2021_\d{5})", re.IGNORECASE)

def normalize_case_layout(root_dir: str) -> str:
    """
    Ensure we have root_dir/BraTS2021_xxxxx/ with *_flair,*_t1,*_t1ce/t1gd,*_t2,*_seg.
    If files are flat, group them into case folders by prefix.
    """
    cases = [d for d in glob.glob(os.path.join(root_dir, "BraTS2021_*")) if os.path.isdir(d)]
    if cases:
        logging.info("Detected %d case folders under %s", len(cases), root_dir)
        return root_dir

    logging.info("No case folders found; attempting to group flat files by case ID...")
    nii = glob.glob(os.path.join(root_dir, "**", "*.nii"), recursive=True)
    nii += glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True)

    grouped = {}
    for f in nii:
        m = CASE_PAT.search(os.path.basename(f))
        if m:
            grouped.setdefault(m.group(1), []).append(f)

    for cid, files in grouped.items():
        dst = os.path.join(root_dir, cid)
        os.makedirs(dst, exist_ok=True)
        for src in files:
            try:
                base = os.path.basename(src)
                if os.path.realpath(os.path.dirname(src)) != os.path.realpath(dst):
                    shutil.move(src, os.path.join(dst, base))
            except shutil.Error:
                pass

    cases = [d for d in glob.glob(os.path.join(root_dir, "BraTS2021_*")) if os.path.isdir(d)]
    logging.info("Grouped into %d case folders", len(cases))
    return root_dir

# -------------------------- IO helpers ------------------------

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def pick(case_dir: str, key_list):
    files = os.listdir(case_dir)
    for k in key_list:
        for f in files:
            if k.lower() in f.lower():
                return os.path.join(case_dir, f)
    raise FileNotFoundError(f"Missing any of {key_list} in {case_dir}")

def load_case_volumes(case_dir: str):
    flair_p = pick(case_dir, ["_flair.nii.gz","_flair.nii"])
    t1_candidates = [f for f in os.listdir(case_dir)
                     if "_t1" in f.lower()
                     and "t1ce" not in f.lower()
                     and "t1gd" not in f.lower()
                     and f.lower().endswith((".nii",".nii.gz"))]
    if not t1_candidates:
        raise FileNotFoundError(f"Plain T1 not found in {case_dir}")
    t1_p   = os.path.join(case_dir, sorted(t1_candidates)[0])
    t1ce_p = pick(case_dir, ["_t1ce.nii.gz","_t1ce.nii","_t1gd.nii.gz","_t1gd.nii"])
    t2_p   = pick(case_dir, ["_t2.nii.gz","_t2.nii"])
    seg_p  = pick(case_dir, ["_seg.nii.gz","_seg.nii"])

    flair = nib.load(flair_p).get_fdata()
    t1    = nib.load(t1_p).get_fdata()
    t1ce  = nib.load(t1ce_p).get_fdata()
    t2    = nib.load(t2_p).get_fdata()
    seg   = nib.load(seg_p).get_fdata()
    return {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2, "seg": seg}

def normalize(slice_):
    s = np.asarray(slice_, dtype=np.float32)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s, dtype=np.uint8)
    nz = finite & (s != 0)
    vals = s[nz] if nz.any() else s[finite]
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(s, dtype=np.uint8)
    p1, p99 = np.percentile(vals, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = vmin, vmax
    s = np.clip(s, p1, p99)
    s = (s - p1) / max(p99 - p1, 1e-6)
    return (s * 255).astype(np.uint8)

def map_labels(seg2d, mode: str):
    s = np.rint(seg2d).astype(np.int16)
    if mode == "binary":
        return (s > 0).astype(np.uint8)
    out = np.zeros_like(s, dtype=np.uint8)
    out[s==1] = 1   # NCR/NET
    out[s==2] = 2   # Edema
    out[s==4] = 3   # Enhancing
    return out

def mask_to_rgb(mask_idx):
    h, w = mask_idx.shape
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    rgb[mask_idx==1] = (0,102,255)  # blue
    rgb[mask_idx==2] = (0,200,0)    # green
    rgb[mask_idx==3] = (255,50,50)  # red
    return rgb

def scan_has_tumor(case_dir: str) -> int:
    try:
        seg_p = pick(case_dir, ["_seg.nii.gz","_seg.nii"])
        seg = nib.load(seg_p).get_fdata()
        return int((seg > 0).any())
    except Exception:
        return 1

# -------------------------- Export logic ----------------------

def export_case(case_dir: str, split: str, out_root: str,
                mask_mode: str, exclude_empty_masks: bool, add_context: int):
    rows = []
    cid = os.path.basename(case_dir)
    vols = load_case_volumes(case_dir)
    flair, t1, t1ce, t2, seg = vols["flair"], vols["t1"], vols["t1ce"], vols["t2"], vols["seg"]
    Z = int(seg.shape[2])

    if exclude_empty_masks:
        tumor_z = [z for z in range(Z) if (seg[:,:,z] > 0).any()]
        keep = set(tumor_z)
        if add_context and tumor_z:
            for z in tumor_z:
                for dz in range(-add_context, add_context+1):
                    zz = z + dz
                    if 0 <= zz < Z: keep.add(zz)
        z_list = sorted(keep)
    else:
        z_list = list(range(Z))

    base_dir = os.path.join(out_root, split, cid)
    flair_dir = os.path.join(base_dir, "flair")
    t1_dir    = os.path.join(base_dir, "t1")
    t1ce_dir  = os.path.join(base_dir, "t1ce")
    t2_dir    = os.path.join(base_dir, "t2")
    msk_dir   = os.path.join(base_dir, "masks")
    ensure_dirs(flair_dir, t1_dir, t1ce_dir, t2_dir, msk_dir)

    for z in z_list:
        try:
            img_flair = normalize(flair[:,:,z])
            img_t1    = normalize(t1[:,:,z])
            img_t1ce  = normalize(t1ce[:,:,z])
            img_t2    = normalize(t2[:,:,z])
            mask_idx  = map_labels(seg[:,:,z], mask_mode)

            # Convert to RGB (replicate single channel 3 times)
            to_rgb = lambda x: np.repeat(x[:, :, None], 3, axis=2)
            img_flair = to_rgb(img_flair)
            img_t1    = to_rgb(img_t1)
            img_t1ce  = to_rgb(img_t1ce)
            img_t2    = to_rgb(img_t2)

            if (mask_idx.max()==0 and
                img_flair.max()==0 and img_t1.max()==0 and
                img_t1ce.max()==0 and img_t2.max()==0):
                continue

            fname = f"z{z:03d}.png"
            p_flair = os.path.join(flair_dir, fname)
            p_t1    = os.path.join(t1_dir,    fname)
            p_t1ce  = os.path.join(t1ce_dir,  fname)
            p_t2    = os.path.join(t2_dir,    fname)
            p_mask  = os.path.join(msk_dir,   fname)
            p_mask_vis = os.path.join(msk_dir, f"z{z:03d}_vis.png")

            imsave(p_flair, img_flair, check_contrast=False)
            imsave(p_t1,    img_t1,    check_contrast=False)
            imsave(p_t1ce,  img_t1ce,  check_contrast=False)
            imsave(p_t2,    img_t2,    check_contrast=False)
            imsave(p_mask,  mask_idx.astype(np.uint8), check_contrast=False)
            imsave(p_mask_vis, mask_to_rgb(mask_idx), check_contrast=False)

            rows.append({
                "split": split, "case_id": cid, "slice_idx": z,
                "flair_path": p_flair, "t1_path": p_t1, "t1ce_path": p_t1ce, "t2_path": p_t2,
                "mask_path": p_mask, "mask_vis_path": p_mask_vis,
                "has_tumor_slice": int(mask_idx.max() > 0), "mask_mode": mask_mode
            })
        except Exception:
            continue
    return rows

# ------------------------------ CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="BraTS 2021 full data prep (extract + per-patient 2D export)")
    ap.add_argument("--src",   required=True, help="Folder containing .tar/.zip OR already-extracted BraTS files")
    ap.add_argument("--extract-dir", required=True, help="Where to place extracted/normalized case folders")
    ap.add_argument("--out-root",    required=True, help="Output root for curated 2D dataset")
    ap.add_argument("--mask-mode",   choices=["binary","3class"], default="3class")
    ap.add_argument("--exclude-empty-masks", action="store_true")
    ap.add_argument("--add-context", type=int, default=0, help="If excluding, also keep ±K neighboring slices")
    ap.add_argument("--train-frac",  type=float, default=0.70)
    ap.add_argument("--val-frac",    type=float, default=0.15)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    setup_logging(args.verbose)
    random.seed(args.seed); np.random.seed(args.seed)

    # 1) Extract if archives present
    if any(name.lower().endswith(ARCHIVE_EXTS) for name in os.listdir(args.src)):
        logging.info("Archives detected in %s — extracting into %s", args.src, args.extract_dir)
        recursive_extract(args.src, args.extract_dir, passes=6)
    else:
        logging.info("No archives in %s; assuming data already extracted.", args.src)
        os.makedirs(args.extract_dir, exist_ok=True)
        if os.path.realpath(args.src) != os.path.realpath(args.extract_dir):
            logging.info("Syncing extracted files into %s ...", args.extract_dir)
            shutil.copytree(args.src, args.extract_dir, dirs_exist_ok=True)

    # 2) Normalize case layout (flat → per-case folders)
    root_cases_dir = normalize_case_layout(args.extract_dir)

    # 3) Discover cases
    case_dirs = sorted([d for d in glob.glob(os.path.join(root_cases_dir, "BraTS2021_*")) if os.path.isdir(d)])
    if not case_dirs:
        raise SystemExit(f"No case folders found under {root_cases_dir}")
    logging.info("Ready cases: %d", len(case_dirs))

    # 4) Build patient-level splits
    meta = []
    for c in tqdm(case_dirs, desc="Scanning cases"):
        meta.append((c, scan_has_tumor(c)))
    pos = [c for c,h in meta if h==1]
    neg = [c for c,h in meta if h==0]
    random.shuffle(pos); random.shuffle(neg)

    def take_frac(lst, frac):
        n = int(round(len(lst)*frac))
        return lst[:n], lst[n:]

    train_pos, rest_pos = take_frac(pos, args.train_frac)
    remaining = 1 - args.train_frac
    val_frac_of_rest = args.val_frac / remaining if remaining > 0 else 0
    val_pos, test_pos = take_frac(rest_pos, val_frac_of_rest)

    train_cases = list(train_pos); val_cases = list(val_pos); test_cases = list(test_pos)
    if neg:
        train_neg, rest_neg = take_frac(neg, args.train_frac)
        val_neg, test_neg   = take_frac(rest_neg, val_frac_of_rest)
        train_cases += train_neg; val_cases += val_neg; test_cases += test_neg

    splits = {"train": train_cases, "val": val_cases, "test": test_cases}
    for s,L in splits.items():
        logging.info("%s: %d cases", s, len(L))

    # 5) Export per-patient, per-modality (multiprocessing)
    os.makedirs(args.out_root, exist_ok=True)
    all_rows = []
    for split, cases in splits.items():
        if not cases:
            continue
        logging.info("Exporting %s (%d cases)...", split, len(cases))
        work = partial(
            export_case,
            split=split, out_root=args.out_root,
            mask_mode=args.mask_mode,
            exclude_empty_masks=args.exclude_empty_masks,
            add_context=max(0, int(args.add_context))
        )
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futures = {ex.submit(work, c): c for c in cases}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split} cases"):
                rows = fut.result()
                if rows: all_rows.extend(rows)

    # 6) Manifest
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out_root, "manifest.csv")
    df.to_csv(csv_path, index=False)
    logging.info("Saved manifest: %s", csv_path)
    if not df.empty:
        stats = df.groupby(["split","has_tumor_slice"]).size().unstack(fill_value=0)
        logging.info("\nSlice counts by split & tumor:\n%s", stats)
        logging.info("Total slices exported: %d", len(df))
    else:
        logging.warning("No slices exported — check inputs/flags.")

if __name__ == "__main__":
    main()