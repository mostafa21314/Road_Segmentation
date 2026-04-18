"""
Step 3: Generate FENet-compatible list/ files for the CULane rural subset.

FENet's culane.py dataset loader reads three fixed index files:
    list/train_gt.txt  ->  /img_path /mask_path lane1 lane2 lane3 lane4
    list/val.txt       ->  /img_path
    list/test.txt      ->  /img_path

The lane existence flags (0 or 1) in train_gt.txt are derived from the
annotation file: if an annotation line exists with >=2 points it counts as
a lane. FENet supports up to 4 lanes (num_classes=5).

Also creates list/test_split/ category files required by evaluate().
Since all images are rural, all test images go into test0_normal.txt and
the other 8 category files are left empty.
"""

from pathlib import Path

DATASET_ROOT = Path("/home/g6/Mostafa/Road_Segmentation/CULane_Rural_Subset(1)/CULane_Rural_Subset")
MAX_LANES = 4


def parse_lane_existence(img_path: Path):
    """Return a list of MAX_LANES existence flags (1/0) by reading the
    annotation file alongside the image."""
    stem = img_path.stem
    # Prefer .lines.txt (after step 1), fall back to .txt
    anno_path = img_path.with_name(stem + ".lines.txt")
    if not anno_path.exists():
        anno_path = img_path.with_suffix(".txt")

    if not anno_path.exists():
        return [0] * MAX_LANES

    lane_count = 0
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split()
            # A valid lane needs at least 2 x,y pairs = 4 values
            if len(values) >= 4:
                lane_count += 1

    # Build existence flags: first lane_count lanes exist, rest don't
    flags = [1 if i < lane_count else 0 for i in range(MAX_LANES)]
    return flags


def generate_train_gt(src_txt: Path, out_path: Path):
    """Generate list/train_gt.txt from the source train.txt list."""
    lines_out = []
    entries = [l.strip() for l in src_txt.read_text().splitlines() if l.strip()]

    for entry in entries:
        # entry: "train/driver_xxx.jpg"
        img_path = DATASET_ROOT / entry
        stem = img_path.stem
        mask_rel = f"/laneseg_label_w16/{img_path.parent.name}/{stem}.png"
        img_rel = f"/{entry}"

        flags = parse_lane_existence(img_path)
        flags_str = " ".join(map(str, flags))
        lines_out.append(f"{img_rel} {mask_rel} {flags_str}")

    out_path.write_text("\n".join(lines_out) + "\n")
    print(f"  Written {len(lines_out)} entries -> {out_path}")


def generate_simple_list(src_txt: Path, out_path: Path):
    """Generate list/val.txt or list/test.txt — just image paths."""
    entries = [l.strip() for l in src_txt.read_text().splitlines() if l.strip()]
    lines_out = [f"/{e}" for e in entries]
    out_path.write_text("\n".join(lines_out) + "\n")
    print(f"  Written {len(lines_out)} entries -> {out_path}")


def generate_test_split(src_txt: Path, split_dir: Path):
    """
    Create list/test_split/ category files.
    All images go into test0_normal.txt (rural subset = normal category).
    test1..test8 are created empty so FENet's evaluate() doesn't crash.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    entries = [l.strip() for l in src_txt.read_text().splitlines() if l.strip()]
    normal_lines = [f"/{e}" for e in entries]

    # Category 0: all test images (rural = normal)
    (split_dir / "test0_normal.txt").write_text("\n".join(normal_lines) + "\n")
    print(f"  test0_normal.txt: {len(normal_lines)} entries")

    # Categories 1–8: empty placeholders
    category_names = [
        "test1_crowd", "test2_hlight", "test3_shadow",
        "test4_noline", "test5_arrow", "test6_curve",
        "test7_cross", "test8_night",
    ]
    for name in category_names:
        (split_dir / f"{name}.txt").write_text("")
    print(f"  {', '.join(n+'.txt' for n in category_names)}: empty placeholders")


def main():
    list_dir = DATASET_ROOT / "list"
    list_dir.mkdir(exist_ok=True)

    # --- train_gt.txt ---
    print("Generating list/train_gt.txt ...")
    generate_train_gt(DATASET_ROOT / "train.txt", list_dir / "train_gt.txt")

    # --- val.txt ---
    print("Generating list/val.txt ...")
    generate_simple_list(DATASET_ROOT / "val.txt", list_dir / "val.txt")

    # --- test.txt ---
    print("Generating list/test.txt ...")
    generate_simple_list(DATASET_ROOT / "test.txt", list_dir / "test.txt")

    # --- test_split/ ---
    print("Generating list/test_split/ category files ...")
    generate_test_split(DATASET_ROOT / "test.txt", list_dir / "test_split")

    print(f"\nDone. List files at: {list_dir}")


if __name__ == "__main__":
    main()
