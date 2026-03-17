import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – save figures without display

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ─────────────────────────────────────────────
# Section 2.1 – Foreground/Background Segmentation
# ─────────────────────────────────────────────

img_pil = Image.open("pillsect.png")
img_color = np.array(img_pil)
gray = np.array(img_pil.convert("L"))

THRESHOLD = 100
binary = (gray > THRESHOLD).astype(np.uint8) * 255

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_color)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Grayscale")
axes[1].axis("off")

axes[2].imshow(binary, cmap="gray", vmin=0, vmax=255)
axes[2].set_title(f"Binary (threshold = {THRESHOLD})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved segmentation.png")

# ─────────────────────────────────────────────
# Section 2.2 – Histogram Analysis
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gray.ravel(), bins=256, range=(0, 256), color="steelblue", edgecolor="none")
ax.axvline(x=THRESHOLD, color="red", linewidth=1.5, label=f"Threshold = {THRESHOLD}")
ax.set_xlabel("Pixel intensity")
ax.set_ylabel("Pixel count")
ax.set_title("Intensity Histogram of pillsect.png (grayscale)")
ax.legend()
plt.tight_layout()
plt.savefig("histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved histogram.png")

# ─────────────────────────────────────────────
# Section 4.1 – Gaussian Convolution via FFT (scale_fft)
# ─────────────────────────────────────────────

def scale_fft(image, sigma):
        
    rows, cols = image.shape
    u = np.fft.fftfreq(cols)          
    v = np.fft.fftfreq(rows)          
    U, V = np.meshgrid(u, v)

    # Frequency-domain Gaussian (FT of spatial Gaussian)
    H = np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2))

    F = np.fft.fft2(image)
    result = np.real(np.fft.ifft2(F * H))
    return result


trui_gray = np.array(Image.open("trui.png").convert("L"), dtype=float)

sigmas = [1, 3, 7, 15]
fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(14, 4))

axes[0].imshow(trui_gray, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

for i, sigma in enumerate(sigmas):
    smoothed = scale_fft(trui_gray, sigma)
    smoothed = np.clip(smoothed, 0, 255)
    axes[i + 1].imshow(smoothed, cmap="gray", vmin=0, vmax=255)
    axes[i + 1].set_title(f"σ = {sigma}")
    axes[i + 1].axis("off")

plt.suptitle("Gaussian Convolution via FFT (scale_fft)", fontsize=13)
plt.tight_layout()
plt.savefig("scale_fft.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved scale_fft.png")
