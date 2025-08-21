# save as augment_all.py

from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from config import config


def _gamma(img: Image.Image, g: float) -> Image.Image:
    inv = 1.0 / g
    lut = [pow(i / 255.0, inv) * 255 for i in range(256)]
    return img.point(lut * 3)


def _noise(img: Image.Image, sigma: float = 3.0) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_imgs(
    train_img_paths,
    train_labels,
    *,
    clean=True,
    resize_to=None  # e.g., (400, 400) or model size like (224, 224)
):
    """
    For each image, apply ALL predefined augmentations (NO rotations, NO erase, NO crop),
    save to <dataset_dir_path>/augmented/<aug_name>/, and return (all_paths, all_labels).
    """
    dataset_root = Path(config.dataset_dir_path)
    out_root = dataset_root / "augmented"

    # clean previous outputs if requested
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # augmentations to apply for EVERY image (crop90, rot_p12, rot_n12, erase_center removed)
    AUGS = [
        ("hflip",        lambda im: ImageOps.mirror(im)),
        ("vflip",        lambda im: ImageOps.flip(im)),
        ("bright_up",    lambda im: ImageEnhance.Brightness(im).enhance(1.12)),
        ("bright_down",  lambda im: ImageEnhance.Brightness(im).enhance(0.88)),
        ("contrast_up",  lambda im: ImageEnhance.Contrast(im).enhance(1.12)),
        ("contrast_down",lambda im: ImageEnhance.Contrast(im).enhance(0.88)),
        ("gamma_up",     lambda im: _gamma(im, 1.05)),
        ("gamma_down",   lambda im: _gamma(im, 0.95)),
        ("blur",         lambda im: im.filter(ImageFilter.GaussianBlur(radius=0.7))),
        ("sharpen",      lambda im: im.filter(ImageFilter.UnsharpMask(radius=1, percent=75, threshold=3))),
        ("noise",        lambda im: _noise(im, sigma=3.0)),
    ]

    # make subfolders
    subdirs = {name: (out_root / name) for name, _ in AUGS}
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)

    all_paths = list(train_img_paths)
    all_labels = list(train_labels)

    for src, lbl in zip(train_img_paths, train_labels):
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping {src}: {e}")
            continue

        # optional standardize size before saving
        if resize_to is not None and img.size != resize_to:
            img = img.resize(resize_to, Image.BILINEAR)

        stem = Path(src).stem

        for aug_name, aug_fn in AUGS:
            out_dir = subdirs[aug_name]
            out_path = out_dir / f"{stem}__{aug_name}.tif"
            try:
                aug = aug_fn(img.copy())
                if resize_to is not None and aug.size != resize_to:
                    aug = aug.resize(resize_to, Image.BILINEAR)
                aug.save(out_path)
                all_paths.append(str(out_path))
                all_labels.append(lbl)
            except Exception as e:
                print(f"[WARN] Could not save {out_path}: {e}")

    return all_paths, all_labels


if __name__ == "__main__":

    # getting the train and test data of fold 0
    from src.prepare_data.prepare_folds import get_folds
    from src.prepare_data.train_test_split import get_img_paths_and_labels
    folds = get_folds()
    train_df, test_df = folds[0]["train_df"], folds[0]["test_df"]
    train_img_path, test_image_path, train_lables, test_lable = get_img_paths_and_labels(train_df, test_df, target_label="Control")

    #augmenting images
    train_img_path, train_lables = augment_imgs(train_img_path, train_lables)
