import os, random, shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
BASE_DIR = Path(__file__).resolve().parent.parent
def split_dataset(
    raw_dir= BASE_DIR / "data/raw",
    out_dir= BASE_DIR /"data/splits",
    train=0.7,
    val=0.15,
    test=0.15,
    seed=42
):
    assert abs((train+val+test) - 1.0) < 1e-6
    random.seed(seed)

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    classes = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    if not classes:
        print("No class folders found in data/raw yet. Create class folders and add images.")
        return

    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        files = [p for p in (raw_dir/cls).iterdir() if p.suffix.lower() in IMG_EXT]
        if len(files) == 0:
            print(f"[WARN] No images in class: {cls}")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = int(n * train)
        n_val = int(n * val)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split, items in splits.items():
            dst = out_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dst / f.name)

        print(f"{cls}: total={n} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

if __name__ == "__main__":
    split_dataset()