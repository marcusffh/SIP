import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import rotate
import os

OUTPUT_DIR = "Assignment_6/output"
BASE = "Assignment_6"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(filename):
    img = Image.open(filename).convert("L")
    return np.array(img, dtype=np.float64) / 255.0


def part_2_1():
    print("=" * 60)
    print("TASK 2.1 — Feature Detection and Image Transforms")
    print("=" * 60)

    img = load_image(f"{BASE}/textlabel_gray_small.png")
    H, W = img.shape
    print(f"Image size: H={H}, W={W}")

    # ── Step 1: Harris corner detection ──────────────────────────────────
    # sigma=3 blurs enough to suppress text noise while keeping card corners.
    # threshold_rel=0.05 keeps the top 5% of corner response values.
    # min_distance=25 ensures we don't get duplicate detections near the same corner.
    harris_resp = corner_harris(img, method="k", k=0.05, sigma=3)
    corners = corner_peaks(harris_resp, min_distance=25, threshold_rel=0.05)
    print(f"Harris detected {len(corners)} corners")

    # ── Step 2: Heuristic — find the 4 label corners via extreme projections ──
    # The 4 card corners are the most extreme points in the image. Text corners
    # are all interior to the card, so they never win any of these extremes.
    #   min(r + c)  →  upper-left  (UL)
    #   max(r + c)  →  lower-right (LR)
    #   min(r - c)  →  upper-right (UR)
    #   max(r - c)  →  lower-left  (LL)
    r = corners[:, 0].astype(float)
    c = corners[:, 1].astype(float)

    p_ul = corners[np.argmin(r + c)]
    p_lr = corners[np.argmax(r + c)]
    p_ur = corners[np.argmin(r - c)]
    p_ll = corners[np.argmax(r - c)]

    print("Label corners (row, col):")
    for name, p in [("UL", p_ul), ("UR", p_ur), ("LL", p_ll), ("LR", p_lr)]:
        print(f"  {name}: row={p[0]}, col={p[1]}")

    # ── Step 3: Estimate rotation from the two long sides ────────────────
    # The label is ~90° CCW from its readable orientation, so its long sides
    # (left: UL→LL, right: UR→LR) run roughly vertically in the image.
    # arctan2(drow, dcol) gives the angle of each side from horizontal.
    # For a nearly vertical side this is ~90°.
    # Heuristic: the label may have a physical notch at one corner (which
    # Harris detects instead of the true geometric corner). We pick the long
    # side whose angle is closest to 90°, since the other side is likely
    # affected by a spurious corner.
    vec_left  = p_ll - p_ul   # UL → LL  (left long side)
    vec_right = p_lr - p_ur   # UR → LR  (right long side)

    angle_left  = np.degrees(np.arctan2(vec_left[0],  vec_left[1]))
    angle_right = np.degrees(np.arctan2(vec_right[0], vec_right[1]))

    # Select the side closest to vertical (90°) as the reliable estimate
    if abs(angle_left - 90) <= abs(angle_right - 90):
        angle_long = angle_left
        print(f"Using LEFT long side (angle {angle_left:.2f}°, dev {abs(angle_left-90):.2f}°)")
    else:
        angle_long = angle_right
        print(f"Using RIGHT long side (angle {angle_right:.2f}°, dev {abs(angle_right-90):.2f}°)")

    print(f"Long-side angle left : {angle_left:.2f}°")
    print(f"Long-side angle right: {angle_right:.2f}°")

    # skimage.transform.rotate: positive = CCW, negative = CW.
    # To make the long side horizontal we rotate CW by angle_long (≈ 90°).
    rotation_angle = -angle_long
    print(f"Applied rotation (skimage): {rotation_angle:.2f}°")

    # ── Step 4: Rotate ───────────────────────────────────────────────────
    rotated = rotate(img, rotation_angle, resize=True)

    # ── Figure 1: All Harris corners + label corners overlaid ────────────
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.scatter(corners[:, 1], corners[:, 0],
               c="red", s=20, marker="+", linewidths=1.0,
               label=f"Harris corners ({len(corners)})")
    label_pts = np.array([p_ul, p_ur, p_ll, p_lr])
    ax.scatter(label_pts[:, 1], label_pts[:, 0],
               c="cyan", s=180, marker="o", zorder=5,
               label="Detected label corners")
    for name, p in [("UL", p_ul), ("UR", p_ur), ("LL", p_ll), ("LR", p_lr)]:
        ax.annotate(name, xy=(p[1] + 6, p[0] - 6),
                    color="yellow", fontsize=10, fontweight="bold")
    ax.set_title("Harris Corners and Detected Label Corners\n"
                 r"($\sigma$=3, $k$=0.05, min_dist=25, thresh_rel=0.05)")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/task2_corners.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/task2_corners.png")

    # ── Figure 2: Original vs rotated ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(rotated, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Rotated by {rotation_angle:.2f}° (CW)")
    axes[1].axis("off")
    plt.suptitle("Task 2.1 — Automatic Label Orientation Correction", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/task2_rotated.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/task2_rotated.png")


if __name__ == "__main__":
    part_2_1()
