import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import random

# MODIFY ONLY THESE PATHS
root = r"path"
output = r"path"
SOURCE = output
DEST = r"path"
MASK_TYPE = "tissue"

# Modalities to slice
modalities = {
    "flair": "_flair.nii.gz",
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "seg": "_seg.nii.gz",
    "tissue": "_csf_gm_wm_tumor.nii.gz",
}


# Helper function: colorize segmentation mask (BraTS)
def colorize_seg(mask: np.ndarray) -> np.ndarray:
    seg_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

    seg_rgb[mask == 1] = [255, 0, 0]       # red
    seg_rgb[mask == 2] = [255, 255, 0]     # yellow
    seg_rgb[mask == 4] = [0, 255, 255]     # cyan

    return seg_rgb


# Helper: colorize tissue mask (CSF / GM / WM / Tumor)
def colorize_tissue(mask: np.ndarray) -> np.ndarray:
    COLORS = {
        0: (0, 0, 0),          # background - black
        1: (0, 0, 255),        # CSF - blue
        2: (0, 255, 0),        # GM - green
        3: (255, 255, 0),      # WM - yellow
        4: (255, 0, 0),        # Tumor - red
        5: (255, 165, 0),      # extra classes if present
        # 6: (255, 0, 255),
        # 7: (0, 255, 255),
        # 8: (128, 0, 128),
        # 9: (128, 128, 0),
        # 10: (0, 128, 128),
        # 11: (128, 0, 0),
        # 12: (0, 128, 0),
    }

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in COLORS.items():
        rgb[mask == label] = color

    return rgb


# Helper: load volume with correct dtype handling
def load_volume(mod_name: str, path: str) -> np.ndarray:
    nii = nib.load(path)
    data = nii.get_fdata()  # always float64 from nibabel

    if mod_name in ["seg", "tissue"]:
        # Round to nearest integer, then cast to uint8 to preserve labels
        data = np.rint(data).astype(np.uint8)
    else:
        # Intensity images: keep as float32
        data = data.astype(np.float32)

    return data


# FUNCTION 1 — SLICE ALL PATIENTS
def slice_all_patients():
    os.makedirs(output, exist_ok=True)
    patients = sorted(os.listdir(root))
    print("Found patients:", len(patients))

    for p in tqdm(patients, desc="Slicing All Patients"):
        p_path = os.path.join(root, p)
        if not os.path.isdir(p_path):
            continue

        patient_out = os.path.join(output, p)
        os.makedirs(patient_out, exist_ok=True)

        loaded_mods = {}

        for mod_name, mod_suffix in modalities.items():
            mod_path = os.path.join(p_path, p + mod_suffix)

            if os.path.exists(mod_path):
                try:
                    vol = load_volume(mod_name, mod_path)
                    loaded_mods[mod_name] = vol
                except Exception as e:
                    print(f"Error loading {mod_name} for {p}: {e}")
            else:
                print(f"Missing {mod_name} for {p}")

        if "flair" not in loaded_mods:
            print("Skipping (no flair):", p)
            continue

        num_slices = loaded_mods["flair"].shape[2]

        for mod in loaded_mods:
            os.makedirs(os.path.join(patient_out, mod), exist_ok=True)

        for s in range(num_slices):
            for mod, volume in loaded_mods.items():
                slice_2d = volume[:, :, s]

                if mod in ["flair", "t1", "t1ce", "t2"]:
                    vmin, vmax = slice_2d.min(), slice_2d.max()
                    if vmax != vmin:
                        norm = (slice_2d - vmin) / (vmax - vmin)
                    else:
                        norm = np.zeros_like(slice_2d)
                    img = (norm * 255).astype(np.uint8)

                elif mod == "seg":
                    mask = slice_2d.astype(np.uint8)
                    img = colorize_seg(mask)

                elif mod == "tissue":
                    mask = slice_2d.astype(np.uint8)
                    img = colorize_tissue(mask)

                else:
                    img = slice_2d.astype(np.uint8)

                save_dir = os.path.join(patient_out, mod)
                save_path = os.path.join(save_dir, f"slice_{s:03}.png")
                Image.fromarray(img).save(save_path)

    print("\n DONE! All modalities + colored masks sliced AXIALLY for all patients!")
    print("Output saved to:", output)


# FUNCTION 2 — REMOVE EMPTY SLICES
def remove_empty_slices():
    root = SOURCE
    modalities = ["flair", "t1", "t1ce", "t2", "seg", "tissue"]

    def is_empty_slice(img_array):
        return np.all(img_array == 0)

    patients = sorted(os.listdir(root))

    for p in tqdm(patients, desc="Checking patients"):
        patient_path = os.path.join(root, p)

        if not os.path.isdir(patient_path):
            continue

        flair_dir = os.path.join(patient_path, "flair")
        if not os.path.exists(flair_dir):
            continue

        slice_files = sorted([f for f in os.listdir(flair_dir) if f.endswith(".png")])

        for slice_file in slice_files:
            slice_empty = True

            for mod in modalities:
                mod_dir = os.path.join(patient_path, mod)
                slice_path = os.path.join(mod_dir, slice_file)

                if not os.path.exists(slice_path):
                    continue

                img = Image.open(slice_path)
                arr = np.array(img)

                if not is_empty_slice(arr):
                    slice_empty = False
                    break

            if slice_empty:
                for mod in modalities:
                    slice_path = os.path.join(patient_path, mod, slice_file)
                    if os.path.exists(slice_path):
                        os.remove(slice_path)

    print("\n DONE! All completely empty slices have been removed.")


# FUNCTION 3 — SPLIT INTO TRAIN/VAL/TEST
def split_dataset():
    SOURCE = output
    DEST = r"path"
    MASK_TYPE = "tissue"

    splits = ["train", "val", "test"]
    for sp in splits:
        os.makedirs(os.path.join(DEST, sp, "flair"), exist_ok=True)
        os.makedirs(os.path.join(DEST, sp, "mask"), exist_ok=True)

    patients = sorted([
        p for p in os.listdir(SOURCE)
        if os.path.isdir(os.path.join(SOURCE, p))
    ])

    print("Total patients:", len(patients))

    random.shuffle(patients)

    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    train_end = int(len(patients) * TRAIN_SPLIT)
    val_end = train_end + int(len(patients) * VAL_SPLIT)

    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]

    splits_dict = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }

    for split_name, patient_list in splits_dict.items():
        for p in tqdm(patient_list, desc=f"Copying {split_name}"):
            flair_dir = os.path.join(SOURCE, p, "flair")
            mask_dir = os.path.join(SOURCE, p, MASK_TYPE)

            for fname in os.listdir(flair_dir):
                src = os.path.join(flair_dir, fname)
                dst = os.path.join(DEST, split_name, "flair", f"{p}_{fname}")
                shutil.copy(src, dst)

            for fname in os.listdir(mask_dir):
                src = os.path.join(mask_dir, fname)
                dst = os.path.join(DEST, split_name, "mask", f"{p}_{fname}")
                shutil.copy(src, dst)

    print("\n Dataset split created with FLAIR + TISSUE MASKS.")
    print("Saved to:", DEST)


# MAIN
def main():
    slice_all_patients()
    remove_empty_slices()
    split_dataset()

if __name__ == "__main__":
    main()
