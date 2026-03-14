import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
import os
from scipy.ndimage import correlate
from skimage import data, io, color
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.ndimage import convolve
from sklearn.cluster import KMeans


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

    # ── Step 2: Find the 4 true label corners from the binary card mask ──
    # The card has a physical fold/notch at one corner. Harris detects the
    # fold as a strong corner feature, so extreme projections on Harris points
    # land on the fold rather than the true card corner.
    # Fix: Otsu-threshold the image to isolate the white card region.
    # A physical concavity (fold) can never win an extreme projection on the
    # white pixel set, so all 4 resulting points are the true card corners.
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

    # ── Step 3: Estimate rotation by averaging the two long sides ────────
    # The label is ~90° CCW from its readable orientation, so its long sides
    # (left: UL→LL, right: UR→LR) run roughly vertically in the image.
    # arctan2(drow, dcol) gives each side's angle from horizontal (~90°).
    # With both corners now correct we can average for a robust estimate.
    vec_left  = p_ll - p_ul   # UL → LL  (left long side)
    vec_right = p_lr - p_ur   # UR → LR  (right long side)

    angle_left  = np.degrees(np.arctan2(vec_left[0],  vec_left[1]))
    angle_right = np.degrees(np.arctan2(vec_right[0], vec_right[1]))
    angle_avg   = (angle_left + angle_right) / 2

    print(f"Long-side angle left : {angle_left:.2f}°")
    print(f"Long-side angle right: {angle_right:.2f}°")
    print(f"Average long-side angle : {angle_avg:.2f}°")

    # skimage.transform.rotate: positive = CCW, negative = CW.
    # To make the long side horizontal we rotate CW by angle_avg (≈ 90°).
    rotation_angle = -angle_avg
    print(f"Applied rotation (skimage): {rotation_angle:.2f}°")

    # ── Step 4: Rotate ───────────────────────────────────────────────────
    rotated = rotate(img, rotation_angle, resize=True)

    # ── Figure 1: Harris corners + true label corners overlaid ───────────
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








#Part 3 #####################################################################################################

# n-jet filter bank up to the third derivatibe
def n_jet(image, sigma):

    responses = {}
    responses["G"] = gaussian_filter(image, sigma=sigma, order=(0,0))
    responses["Gx"] = gaussian_filter(image, sigma=sigma, order=(0,1))
    responses["Gy"] = gaussian_filter(image, sigma=sigma, order=(1,0))
    responses["Gxx"] = gaussian_filter(image, sigma=sigma, order=(0,2))
    responses["Gxy"] = gaussian_filter(image, sigma=sigma, order=(1,1))
    responses["Gyy"] = gaussian_filter(image, sigma=sigma, order=(2,0))
    responses["Gxxx"] = gaussian_filter(image, sigma=sigma, order=(0,3))
    responses["Gxxy"] = gaussian_filter(image, sigma=sigma, order=(1,2))
    responses["Gxyy"] = gaussian_filter(image, sigma=sigma, order=(2,1))
    responses["Gyyy"] = gaussian_filter(image, sigma=sigma, order=(3,0))

    #returns a dictionary:)
    return responses


# impulse generator for visualization
def create_impulse(size=31):
    img = np.zeros((size, size)) 
    center = size // 2
    img[center, center] = 1
    return img

def save_njet_grid(responses, sigma, output_dir, prefix="filter", cmap="seismic"):

    #create output directory
    os.makedirs(output_dir, exist_ok=True)

    #create filter and plot axes (2 times 5 images)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5)) 

    for ax, (name, img) in zip(axes.flat, responses.items()):
        v = np.max(np.abs(img))
        ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
        ax.set_title(fr"{name}  $\sigma=${sigma}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{prefix}_njet_sigma{sigma}.png")
    plt.savefig(filepath, dpi=150)

    plt.close()





def part_3_1():

    #load image in greyscale
    img = load_image("input/sunandsea.jpg")
    impulse = create_impulse(size= 100)

    #use the filter
    img = n_jet(img, 5)
    impulse = n_jet(impulse, 5)

    save_njet_grid(img, sigma= 5, output_dir="output", prefix="3_1_image_njet",cmap= "grey" )
    save_njet_grid(impulse, sigma= 5, output_dir="output", prefix="3_1_impulse_njet",cmap= "grey" )


def learn_pca_filterbank(image, patch_size=8, n_filters=16, max_patches=10000):
    # Extract patches and flatten
    patches = extract_patches_2d(image, (patch_size, patch_size), max_patches=max_patches, random_state=42)
    X = patches.reshape(len(patches), -1)
    
    # Fit PCA
    pca = PCA(n_components=n_filters)
    pca.fit(X)
    
    # Reshape each component back to 2D
    filters = pca.components_.reshape(n_filters, patch_size, patch_size)
    
    return filters  



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

    filepath = os.path.join(output_dir, f"3_2_{prefix}_pca_filterbank.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

def part_3_2():
    # Load image
    image = load_image("input/sunandsea.jpg")
    impulse = create_impulse(size=8)

    #learn filters from image
    filters = learn_pca_filterbank(image, n_filters=64)

    # visualize filters using impulse image
    filter_responses = apply_pca_filterbank(impulse, filters, n=64)
    pca_filterbank_plot(filter_responses, "output", prefix="filters", cmap="grey")

    #apply to actual image and visualize
    image_responses = apply_pca_filterbank(image, filters, n=64)
    pca_filterbank_plot(image_responses, "output", prefix="responses", cmap="grey")





def responses_to_feature_matrix(responses):
    stacked = np.stack(list(responses.values())) #H x W x channels(features)
    X = stacked.reshape(len(responses), -1).T #pixels x channels(feature)
    return X

def reduce_features(X, n_components=3):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

def kmeans_segment(X, image_shape, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    segmentation = kmeans.labels_.reshape(image_shape)
    return segmentation


def save_segmentation(segmentation, output_dir, prefix="segmentation"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(segmentation, cmap='tab10')
    plt.title(f"{prefix} (K=3)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"3_3{prefix}.png"), dpi=150, bbox_inches ='tight')
    plt.close()



def part_3_3():
    image = load_image("input/sunandsea.jpg")

    # PCA filterbank
    pca_filters = learn_pca_filterbank(image, n_filters=64) # learn filters
    pca_responses = apply_pca_filterbank(image, pca_filters, n=3) # apply filters
    pca_X = responses_to_feature_matrix(pca_responses) # convert to feature matrix from dictionary
    pca_segmented = kmeans_segment(pca_X, image.shape, n_clusters=3) #segment
    save_segmentation(pca_segmented, "output", prefix="pca_segmentation") #save segmentation

    # N-Jet
    njet_responses = n_jet(image, sigma=5) #apply filters
    njet_X = responses_to_feature_matrix(njet_responses) # convert to feature matrix from dictionary
    njet_X_reduced = reduce_features(njet_X, n_components=3) #reduce features using pca
    njet_segmented = kmeans_segment(njet_X_reduced, image.shape, n_clusters=3) #segment
    save_segmentation(njet_segmented, "output", prefix="N-jet_segmentation")# save it



    



    


if __name__ == "__main__":
    #part_2_1()
    part_3_1()
    part_3_2()
    part_3_3()
