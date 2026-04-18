"""
Step 1: Rename CULane annotation files from *.txt to *.lines.txt

FENet's dataset loader (fenet/datasets/culane.py) looks for annotation files as:
    img_path[:-3] + 'lines.txt'
i.e. for driver_xxx.jpg it expects driver_xxx.lines.txt

This script renames annotation .txt files inside the train/, test/, val/
subdirectories of the curated dataset. Root-level list files (train.txt,
test.txt, val.txt) are NOT touched — they live at the dataset root, not
inside a split subdir.
"""

from pathlib import Path

DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
SPLITS = ["train", "test", "val"]


def rename_split(split_dir: Path):
    renamed = 0
    skipped = 0
    for txt_file in sorted(split_dir.glob("*.txt")):
        jpg_sibling = txt_file.with_suffix(".jpg")
        if jpg_sibling.exists():
            new_path = txt_file.with_name(txt_file.stem + ".lines.txt")
            txt_file.rename(new_path)
            renamed += 1
        else:
            print(f"  [SKIP] No matching .jpg for: {txt_file.name}")
            skipped += 1
    return renamed, skipped


def main():
    total_renamed = 0
    total_skipped = 0

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.is_dir():
            print(f"[WARN] Directory not found, skipping: {split_dir}")
            continue

        print(f"Processing {split}/...")
        renamed, skipped = rename_split(split_dir)
        print(f"  Renamed: {renamed}  |  Skipped (no .jpg match): {skipped}")
        total_renamed += renamed
        total_skipped += skipped

    print(f"\nDone. Total renamed: {total_renamed}  |  Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
