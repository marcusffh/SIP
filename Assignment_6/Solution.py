import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import rotate
import os
from scipy.ndimage import correlate
from skimage import data


###### 1.4 #########
# odd image size
N = 101

# create zero image
img = np.zeros((N, N))

# center coordinates
c = N // 2

# size of white square
s = 10

# insert white square
img[c-s:c+s+1, c-s:c+s+1] = 1

plt.imshow(img, cmap='gray')
plt.title("Zero image with centered white square")
plt.axis("off")
plt.show()


###### 1.5 #########

def translate_integer_filter(img, tx, ty, mode='constant', cval=0.0):
    """
    Translate an image by integer amounts (tx, ty) using a filter mask

    img : input 2D image
    tx :  translation in x-direction Positive = right
    ty :  translation in y-direction Positive = down
    mode : boundary condition passed to scipy.ndimage.correlate e.g.  'constant', 'nearest', 'reflect', 'wrap'
    cval : constant value used when mode='constant'

    output : translated image
    """
    if not isinstance(tx, int) or not isinstance(ty, int):
        raise ValueError("tx and ty must be integers.")

    # Kernel size large enough to encode the shift
    h = 2 * abs(ty) + 1
    w = 2 * abs(tx) + 1
    kernel = np.zeros((h, w), dtype=float)

    cy = abs(ty)
    cx = abs(tx)

    # Place the 1 so that correlation gives: out(y, x) = img(y - ty, x - tx)
    kernel[cy - ty, cx - tx] = 1.0

    out = correlate(img, kernel, mode=mode, cval=cval)
    return out

# apply translations
img_right = translate_integer_filter(img, tx=15, ty=0, mode='constant', cval=0)
img_down  = translate_integer_filter(img, tx=0, ty=12, mode='constant', cval=0)
img_diag  = translate_integer_filter(img, tx=10, ty=8, mode='constant', cval=0)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Original")
axes[1].imshow(img_right, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("tx=15, ty=0")
axes[2].imshow(img_down, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("tx=0, ty=12")
axes[3].imshow(img_diag, cmap='gray', vmin=0, vmax=1)
axes[3].set_title("tx=10, ty=8")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


########## 1.6 ############

def translate_nearest(img, tx, ty):
    """
    Translate a 2D image by (tx, ty) using backward mapping
    and nearest neighbor interpolation

    img : input 2D image
    tx : translation in x-direction Positive = right.
    ty : translation in y-direction Positive = down.

    output : translated image
    """
    H, W = img.shape
    out = np.zeros_like(img)

    for y_prime in range(H):
        for x_prime in range(W):
            # inverse mapping
            x = x_prime - tx
            y = y_prime - ty

            # nearest neighbor interpolation
            x_nn = int(round(x))
            y_nn = int(round(y))

            # check bounds
            if 0 <= x_nn < W and 0 <= y_nn < H:
                out[y_prime, x_prime] = img[y_nn, x_nn]

    return out

translated = translate_nearest(img, tx=0.6, ty=1.2)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(translated, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(r"Translated image, $t=(0.6,\,1.2)^T$")
axes[1].axis("off")

plt.tight_layout()
plt.show()


######### 1.7 ###########

def translate_fourier(img, tx, ty):
    """
    Translate an image using the Fourier shift theorem
    """

    H, W = img.shape

    # fourier transform
    F = np.fft.fft2(img)

    # frequency coordinates
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    U, V = np.meshgrid(u, v)

    # phase shift
    phase = np.exp(-2j * np.pi * (U * tx + V * ty))

    # apply shift in frequency domain
    F_shifted = F * phase

    # inverse transform
    shifted = np.fft.ifft2(F_shifted)

    return np.real(shifted)

translated_fft = translate_fourier(img, 10, 15)

fig, ax = plt.subplots(1,3,figsize=(12,4))

ax[0].imshow(img, cmap='gray', vmin=0, vmax=1)
ax[0].set_title("Original")

ax[1].imshow(translated, cmap='gray', vmin=0, vmax=1)
ax[1].set_title("Nearest neighbor")

ax[2].imshow(translated_fft, cmap='gray', vmin=0, vmax=1)
ax[2].set_title("Fourier translation")

for a in ax:
    a.axis("off")

plt.show()


####### 1.8 ##########

translated_fft = translate_fourier(img, 1.2, 0.6)

fig, ax = plt.subplots(1,3,figsize=(12,4))

ax[1].imshow(translated_fft, cmap='gray', vmin=0, vmax=1)
ax[1].set_title("Fourier translation")

for a in ax:
    a.axis("off")

plt.show()

img2 = data.camera()
translated2 = translate_fourier(img2, 0.6, 1.2)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img2, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(translated2, cmap='gray')
plt.title("Fourier translation")

plt.show()

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
