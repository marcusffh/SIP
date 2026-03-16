import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter, correlate, convolve
from skimage import data
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(filename):
    img = Image.open(filename).convert("L")
    return np.array(img, dtype=np.float64) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1
# ─────────────────────────────────────────────────────────────────────────────

def _make_square_img(N=101, s=10):
    """Create a zero image with a centered white square."""
    img = np.zeros((N, N))
    c = N // 2
    img[c-s:c+s+1, c-s:c+s+1] = 1
    return img


def part_1_4():
    print("=" * 60)
    print("TASK 1.4 — Zero image with centered white square")
    print("=" * 60)
    img = _make_square_img()
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Zero image with centered white square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_4_white_square.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_4_white_square.png")


def translate_integer_filter(img, tx, ty, mode='constant', cval=0.0):
    """Translate image by integer (tx, ty) using a filter mask."""
    if not isinstance(tx, int) or not isinstance(ty, int):
        raise ValueError("tx and ty must be integers.")
    h = 2 * abs(ty) + 1
    w = 2 * abs(tx) + 1
    kernel = np.zeros((h, w), dtype=float)
    cy, cx = abs(ty), abs(tx)
    kernel[cy - ty, cx - tx] = 1.0
    return correlate(img, kernel, mode=mode, cval=cval)


def part_1_5():
    print("=" * 60)
    print("TASK 1.5 — Integer translation via filter mask")
    print("=" * 60)
    img = _make_square_img()
    img_right = translate_integer_filter(img, tx=15, ty=0)
    img_down  = translate_integer_filter(img, tx=0,  ty=12)
    img_diag  = translate_integer_filter(img, tx=10, ty=8)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, image, title in zip(axes,
                                [img, img_right, img_down, img_diag],
                                ["Original", "tx=15, ty=0", "tx=0, ty=12", "tx=10, ty=8"]):
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_5_translations.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_5_translations.png")


def translate_nearest(img, tx, ty):
    """Translate image by (tx, ty) using backward mapping + nearest-neighbour."""
    H, W = img.shape
    out = np.zeros_like(img)
    for y_prime in range(H):
        for x_prime in range(W):
            x_nn = int(round(x_prime - tx))
            y_nn = int(round(y_prime - ty))
            if 0 <= x_nn < W and 0 <= y_nn < H:
                out[y_prime, x_prime] = img[y_nn, x_nn]
    return out


def part_1_6():
    print("=" * 60)
    print("TASK 1.6 — Sub-pixel translation via nearest-neighbour")
    print("=" * 60)
    img = _make_square_img()
    translated = translate_nearest(img, tx=0.6, ty=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    axes[1].imshow(translated, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(r"Nearest-neighbour $t=(0.6,\,1.2)^T$")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_6_nearest_neighbour.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_6_nearest_neighbour.png")


def translate_fourier(img, tx, ty):
    """Translate an image using the Fourier shift theorem."""
    H, W = img.shape
    F = np.fft.fft2(img)
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    U, V = np.meshgrid(u, v)
    phase = np.exp(-2j * np.pi * (U * tx + V * ty))
    return np.real(np.fft.ifft2(F * phase))


def part_1_7():
    print("=" * 60)
    print("TASK 1.7 — Translation via Fourier shift theorem")
    print("=" * 60)
    img = _make_square_img()
    translated_nn  = translate_nearest(img, tx=0.6, ty=1.2)
    translated_fft = translate_fourier(img, tx=10, ty=15)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, image, title in zip(ax,
                               [img, translated_nn, translated_fft],
                               ["Original", "Nearest-neighbour (0.6, 1.2)", "Fourier (10, 15)"]):
        a.imshow(image, cmap='gray', vmin=0, vmax=1)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_7_fourier_translation.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_7_fourier_translation.png")


def part_1_8():
    print("=" * 60)
    print("TASK 1.8 — Sub-pixel Fourier translation on real image")
    print("=" * 60)
    img = _make_square_img()
    translated_sq = translate_fourier(img, 1.2, 0.6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original (square)")
    axes[0].axis("off")
    axes[1].imshow(translated_sq, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(r"Fourier $t=(1.2,\,0.6)^T$")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_8_fourier_subpixel_square.png", dpi=150)
    plt.close()

    img2 = data.camera().astype(np.float64) / 255.0
    translated2 = translate_fourier(img2, 0.6, 1.2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img2, cmap='gray')
    axes[0].set_title("Original (camera)")
    axes[0].axis("off")
    axes[1].imshow(translated2, cmap='gray')
    axes[1].set_title(r"Fourier $t=(0.6,\,1.2)^T$")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_8_fourier_subpixel_camera.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_8_fourier_subpixel_square.png")
    print(f"Saved: {OUTPUT_DIR}/1_8_fourier_subpixel_camera.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2
# ─────────────────────────────────────────────────────────────────────────────

def part_2_1():
    print("=" * 60)
    print("TASK 2.1 — Feature Detection and Image Transforms")
    print("=" * 60)

    img = load_image(f"{BASE}/textlabel_gray_small.png")
    H, W = img.shape
    print(f"Image size: H={H}, W={W}")

    harris_resp = corner_harris(img, method="k", k=0.05, sigma=3)
    corners = corner_peaks(harris_resp, min_distance=25, threshold_rel=0.05)
    print(f"Harris detected {len(corners)} corners")

    from skimage.filters import threshold_otsu
    t_otsu = threshold_otsu(img)
    mask = img > t_otsu
    ys, xs = np.where(mask)
    print(f"Otsu threshold: {t_otsu:.3f}")

    rm, cm = ys.astype(float), xs.astype(float)
    p_ul = np.array([ys[np.argmin(rm + cm)], xs[np.argmin(rm + cm)]])
    p_lr = np.array([ys[np.argmax(rm + cm)], xs[np.argmax(rm + cm)]])
    p_ur = np.array([ys[np.argmin(rm - cm)], xs[np.argmin(rm - cm)]])
    p_ll = np.array([ys[np.argmax(rm - cm)], xs[np.argmax(rm - cm)]])

    print("Label corners (row, col):")
    for name, p in [("UL", p_ul), ("UR", p_ur), ("LL", p_ll), ("LR", p_lr)]:
        print(f"  {name}: row={p[0]}, col={p[1]}")

    vec_left  = p_ll - p_ul
    vec_right = p_lr - p_ur
    angle_left  = np.degrees(np.arctan2(vec_left[0],  vec_left[1]))
    angle_right = np.degrees(np.arctan2(vec_right[0], vec_right[1]))
    angle_avg   = (angle_left + angle_right) / 2
    rotation_angle = -angle_avg

    print(f"Long-side angle left : {angle_left:.2f}°")
    print(f"Long-side angle right: {angle_right:.2f}°")
    print(f"Applied rotation     : {rotation_angle:.2f}°")

    rotated = rotate(img, rotation_angle, resize=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3
# ─────────────────────────────────────────────────────────────────────────────

def n_jet(image, sigma):
    """N-jet filter bank up to third-order Gaussian derivatives."""
    responses = {}
    for order, name in [
        ((0, 0), "G"),
        ((0, 1), "Gx"),   ((1, 0), "Gy"),
        ((0, 2), "Gxx"),  ((1, 1), "Gxy"),  ((2, 0), "Gyy"),
        ((0, 3), "Gxxx"), ((1, 2), "Gxxy"), ((2, 1), "Gxyy"), ((3, 0), "Gyyy"),
    ]:
        responses[name] = gaussian_filter(image, sigma=sigma, order=order)
    return responses


def create_impulse(size=31):
    img = np.zeros((size, size))
    img[size // 2, size // 2] = 1
    return img


def save_njet_grid(responses, sigma, output_dir, prefix="filter", cmap="seismic"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, (name, img) in zip(axes.flat, responses.items()):
        v = np.max(np.abs(img))
        ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
        ax.set_title(fr"{name}  $\sigma=${sigma}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_njet_sigma{sigma}.png"), dpi=150)
    plt.close()


def part_3_1():
    print("=" * 60)
    print("TASK 3.1 — N-jet filter bank")
    print("=" * 60)
    img     = load_image(f"{BASE}/input/sunandsea.jpg")
    impulse = create_impulse(size=100)

    save_njet_grid(n_jet(img, 5),     sigma=5, output_dir=OUTPUT_DIR, prefix="3_1_image_njet",   cmap="grey")
    save_njet_grid(n_jet(impulse, 5), sigma=5, output_dir=OUTPUT_DIR, prefix="3_1_impulse_njet", cmap="grey")
    print(f"Saved N-jet grids to {OUTPUT_DIR}/")


def learn_pca_filterbank(image, patch_size=8, n_filters=16, max_patches=10000):
    patches = extract_patches_2d(image, (patch_size, patch_size),
                                 max_patches=max_patches, random_state=42)
    X = patches.reshape(len(patches), -1)
    pca = PCA(n_components=n_filters)
    pca.fit(X)
    return pca.components_.reshape(n_filters, patch_size, patch_size)


def apply_pca_filterbank(image, filters, n=1):
    responses = {}
    for i in range(n):
        responses[f"PC{i+1}"] = convolve(image, filters[i], mode='reflect')
    return responses


def pca_filterbank_plot(responses, output_dir, prefix="filter", cmap="seismic"):
    n_filters = len(responses)
    n_rows = n_filters // 8
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(n_rows, 8, figsize=(12, n_rows * 2))
    for ax, (name, img) in zip(axes.flat, responses.items()):
        v = np.max(np.abs(img))
        ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"3_2_{prefix}_pca_filterbank.png"), dpi=150)
    plt.close()


def part_3_2():
    print("=" * 60)
    print("TASK 3.2 — PCA filter bank")
    print("=" * 60)
    image   = load_image(f"{BASE}/input/sunandsea.jpg")
    impulse = create_impulse(size=8)
    filters = learn_pca_filterbank(image, n_filters=64)

    pca_filterbank_plot(apply_pca_filterbank(impulse, filters, n=64),
                        OUTPUT_DIR, prefix="filters", cmap="grey")
    pca_filterbank_plot(apply_pca_filterbank(image, filters, n=64),
                        OUTPUT_DIR, prefix="responses", cmap="grey")
    print(f"Saved PCA filterbank plots to {OUTPUT_DIR}/")


def responses_to_feature_matrix(responses):
    stacked = np.stack(list(responses.values()))
    return stacked.reshape(len(responses), -1).T


def reduce_features(X, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def kmeans_segment(X, image_shape, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_.reshape(image_shape)


def save_segmentation(segmentation, output_dir, prefix="segmentation"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(segmentation, cmap='tab10')
    plt.title(f"{prefix} (K=3)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"3_3{prefix}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def part_3_3():
    print("=" * 60)
    print("TASK 3.3 — Segmentation via filter-bank features + K-means")
    print("=" * 60)
    image = load_image(f"{BASE}/input/sunandsea.jpg")

    # PCA filterbank path
    pca_filters    = learn_pca_filterbank(image, n_filters=64)
    pca_responses  = apply_pca_filterbank(image, pca_filters, n=3)
    pca_X          = responses_to_feature_matrix(pca_responses)
    pca_segmented  = kmeans_segment(pca_X, image.shape, n_clusters=3)
    save_segmentation(pca_segmented, OUTPUT_DIR, prefix="pca_segmentation")

    # N-jet path
    njet_responses = n_jet(image, sigma=5)
    njet_X         = responses_to_feature_matrix(njet_responses)
    njet_X_reduced = reduce_features(njet_X, n_components=3)
    njet_segmented = kmeans_segment(njet_X_reduced, image.shape, n_clusters=3)
    save_segmentation(njet_segmented, OUTPUT_DIR, prefix="N-jet_segmentation")

    print(f"Saved segmentation maps to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    part_1_4()
    part_1_5()
    part_1_6()
    part_1_7()
    part_1_8()
    part_2_1()
    part_3_1()
    part_3_2()
    part_3_3()
