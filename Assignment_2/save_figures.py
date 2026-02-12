"""
Save all report figures for Assignment 2 (Parts 3.2, 3.3, 4.1–4.4).
Run from the Assignment_2 directory.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.filters import gaussian
import os

out = 'report_figures'
os.makedirs(out, exist_ok=True)

# ============================================================
# 3.2 — Gaussian filter with fixed sigma=5, increasing kernel size N
# ============================================================
eight = img_as_float(imread('eight.tif'))
if eight.ndim == 3:
    eight = rgb2gray(eight)

eight_noisy = random_noise(eight, mode='gaussian', var=0.01)

sigma = 5
kernel_sizes = [3, 5, 9, 15, 31]

fig, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(20, 4))
axes[0].imshow(eight_noisy, cmap='gray')
axes[0].set_title('Noisy image')
axes[0].axis('off')

for i, N in enumerate(kernel_sizes):
    truncate = (N - 1) / (2 * sigma)
    filtered = gaussian(eight_noisy, sigma=sigma, truncate=truncate)
    axes[i + 1].imshow(filtered, cmap='gray')
    axes[i + 1].set_title(f'N = {N}')
    axes[i + 1].axis('off')

plt.suptitle(f'Gaussian filter with σ = {sigma}, increasing kernel size N', fontsize=14)
plt.tight_layout()
plt.savefig(f'{out}/3_2_gaussian_fixed_sigma_increasing_N.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 3_2_gaussian_fixed_sigma_increasing_N.png')

# ============================================================
# 3.3 — Gaussian filter with increasing sigma, N = 3*sigma
# ============================================================
sigmas = [1, 2, 3, 5, 8]

fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(20, 4))
axes[0].imshow(eight_noisy, cmap='gray')
axes[0].set_title('Noisy image')
axes[0].axis('off')

for i, s in enumerate(sigmas):
    N = int(3 * s)
    truncate = (N - 1) / (2 * s) if s > 0 else 0
    filtered = gaussian(eight_noisy, sigma=s, truncate=max(truncate, 0.1))
    axes[i + 1].imshow(filtered, cmap='gray')
    axes[i + 1].set_title(f'σ = {s}, N = {N}')
    axes[i + 1].axis('off')

plt.suptitle('Gaussian filter with increasing σ (N = 3σ)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{out}/3_3_gaussian_increasing_sigma.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 3_3_gaussian_increasing_sigma.png')

# ============================================================
# 4.1 — CDF of pout.tif
# ============================================================
def compute_cdf(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    return cdf, bins

pout = imread('pout.tif')
if pout.ndim == 3:
    pout = rgb2gray(pout)
    pout = img_as_ubyte(pout)

cdf_pout, bins_pout = compute_cdf(pout)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(pout, cmap='gray')
axes[0].set_title('pout.tif')
axes[0].axis('off')
axes[1].plot(range(256), cdf_pout)
axes[1].set_xlabel('Intensity')
axes[1].set_ylabel('CDF')
axes[1].set_title('Cumulative Distribution Function of pout.tif')
axes[1].set_xlim([0, 255])
plt.tight_layout()
plt.savefig(f'{out}/4_1_cdf_pout.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 4_1_cdf_pout.png')

# ============================================================
# 4.2 — CDF mapping C(I) on pout.tif
# ============================================================
def apply_cdf(image, cdf):
    return cdf[image]

pout_equalized = apply_cdf(pout, cdf_pout)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(pout, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original pout.tif')
axes[0].axis('off')
axes[1].imshow(pout_equalized, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('After CDF mapping C(I)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(f'{out}/4_2_cdf_mapping.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 4_2_cdf_mapping.png')

# ============================================================
# 4.4 — Histogram matching (pout.tif → eight.tif)
# ============================================================
def pseudo_inverse_cdf(cdf, l):
    indices = np.where(cdf >= l)[0]
    return int(np.min(indices))

def histogram_match(I1, I2):
    cdf1, _ = compute_cdf(I1)
    cdf2, _ = compute_cdf(I2)
    lookup = np.zeros(256, dtype=np.uint8)
    for s in range(256):
        l = cdf1[s]
        lookup[s] = pseudo_inverse_cdf(cdf2, l)
    J = lookup[I1]
    return J

eight_gray = imread('eight.tif')
if eight_gray.ndim == 3:
    eight_gray = rgb2gray(eight_gray)
    eight_gray = img_as_ubyte(eight_gray)

matched = histogram_match(pout, eight_gray)

# Images
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(pout, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Source: pout.tif')
axes[0].axis('off')
axes[1].imshow(eight_gray, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Target: eight.tif')
axes[1].axis('off')
axes[2].imshow(matched, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Matched result')
axes[2].axis('off')
plt.tight_layout()
plt.savefig(f'{out}/4_4_histogram_matching_images.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 4_4_histogram_matching_images.png')

# CDF comparison
cdf_source, _ = compute_cdf(pout)
cdf_target, _ = compute_cdf(eight_gray)
cdf_matched, _ = compute_cdf(matched)

plt.figure(figsize=(8, 5))
plt.plot(range(256), cdf_source, label='Source (pout.tif)')
plt.plot(range(256), cdf_target, label='Target (eight.tif)')
plt.plot(range(256), cdf_matched, '--', label='Matched result')
plt.xlabel('Intensity')
plt.ylabel('CDF')
plt.title('Cumulative histograms comparison')
plt.legend()
plt.xlim([0, 255])
plt.savefig(f'{out}/4_4_histogram_matching_cdfs.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 4_4_histogram_matching_cdfs.png')

print('\nAll figures saved to report_figures/')
