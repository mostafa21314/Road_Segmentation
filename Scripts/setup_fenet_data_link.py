"""
Step 4: Create the data/CULane symlink inside the FENet repo.

FENet's config hardcodes:
    dataset_path = './data/CULane'

So it always looks for the dataset at FENet/data/CULane/ relative to where
you launch main.py from. Rather than copying the dataset, we create a symlink:
    FENet/data/CULane  ->  CULane_Rural_Subset(1)/CULane_Rural_Subset

This script is safe to re-run: it will report if the link already exists
and is correct, or replace a broken/wrong link after confirmation.
"""

import os
from pathlib import Path

FENET_ROOT   = Path("/home/g6/Mostafa/Road_Segmentation/FENet")
DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
LINK_PATH    = FENET_ROOT / "data" / "CULane"


def main():
    print(f"FENet root  : {FENET_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Symlink path: {LINK_PATH}")
    print()

    # Sanity check — dataset root must exist
    if not DATASET_ROOT.is_dir():
        print(f"[ERROR] Dataset root does not exist: {DATASET_ROOT}")
        return

    # Create FENet/data/ if it doesn't exist
    LINK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Handle existing path at the link location
    if LINK_PATH.exists() or LINK_PATH.is_symlink():
        if LINK_PATH.is_symlink():
            current_target = Path(os.readlink(LINK_PATH))
            if current_target == DATASET_ROOT or LINK_PATH.resolve() == DATASET_ROOT.resolve():
                print("[OK] Symlink already exists and points to the correct dataset.")
                return
            else:
                print(f"[WARN] Symlink exists but points to a different target: {current_target}")
                answer = input("Replace it? [y/N]: ").strip().lower()
                if answer != "y":
                    print("Aborted.")
                    return
                LINK_PATH.unlink()
        else:
            print(f"[ERROR] {LINK_PATH} exists and is not a symlink. Remove it manually first.")
            return

    # Create the symlink
    LINK_PATH.symlink_to(DATASET_ROOT)
    print(f"[OK] Created symlink: {LINK_PATH} -> {DATASET_ROOT}")

    # Quick verification
    required = ["list/train_gt.txt", "list/val.txt", "list/test.txt", "laneseg_label_w16"]
    print("\nVerifying expected dataset structure through symlink:")
    all_ok = True
    for rel in required:
        target = LINK_PATH / rel
        status = "OK" if target.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  [{status}] data/CULane/{rel}")

    if all_ok:
        print("\nAll checks passed. You can now run FENet from its root:")
        print("  cd", FENET_ROOT)
        print("  python main.py configs/fenet/FENetV2_dla34_culane.py --load_from ./fenetv2_culane_dla34.pth --gpus 0")
    else:
        print("\n[WARN] Some items are missing — make sure steps 1-3 have been run first.")


if __name__ == "__main__":
    main()
