"""
Assignment 5 – Features and Scale-Space Theory
Signal and Image Processing
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, fftfreq
from scipy.signal import stft as scipy_stft
import subprocess
from scipy.ndimage import gaussian_laplace
from skimage.feature import canny, corner_harris, corner_peaks, peak_local_max
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE = os.path.dirname(__file__)


def load_image(filename):
    """Load a grayscale image as a float64 numpy array."""
    path = os.path.join(BASE, filename)
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def gaussian_kernel_2d(size, sigma):
    """Create a normalized 2D Gaussian kernel of given size."""
    half = size // 2
    x = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, x)
    k = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return k / k.sum()


def embed_kernel(kernel, shape):
    """
    Embed kernel in array of given shape with kernel center at (0,0).
    Required for correct circular convolution via FFT.
    """
    M, N = shape
    kh, kw = kernel.shape
    k_embed = np.zeros((M, N))
    k_embed[:kh, :kw] = kernel
    k_embed = np.roll(k_embed, -(kh // 2), axis=0)
    k_embed = np.roll(k_embed, -(kw // 2), axis=1)
    return k_embed


def gaussian_2d(X, Y, sigma):
    """Evaluate 2D Gaussian G(x,y,sigma) at coordinates X, Y."""
    return (1.0 / (2 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))


# =============================================================================
# Part 1.1 – Spectrogram of progression.wav
# =============================================================================

def compute_spectrogram(audio, sample_rate, window_size):
    """
    Compute STFT magnitude, time axis, and frequency axis.
    Uses scipy.signal.stft with Hann window and 50% overlap.
    """
    stride = window_size // 2
    f, t, Zxx = scipy_stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=window_size,
        noverlap=window_size - stride,
    )
    magnitude = np.abs(Zxx)
    return magnitude, t, f


def part_1_1():
    print("=" * 60)
    print("Part 1.1: Spectrogram of progression.wav")
    print("=" * 60)

    path = os.path.join(BASE, "progression.wav")
    sample_rate = 44100  # known from ffprobe

    # The file is AAC-in-M4A despite the .wav extension; decode via ffmpeg
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64)

    window_size = 2048
    magnitude, t, f = compute_spectrogram(audio, sample_rate, window_size)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        np.log1p(magnitude),
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[-1]],
        cmap="inferno",
    )
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram of progression.wav  (Hann window, N={window_size}, stride={window_size//2})")
    fig.colorbar(im, ax=ax, label="Log Magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_1_1.png"), dpi=150)
    plt.close()
    print(f"  fs={sample_rate} Hz, window_size={window_size}, stride={window_size//2}")
    print("  Saved part_1_1.png")
    print()


# =============================================================================
# Part 2.1 – LSI Degradation Model
# =============================================================================

def apply_lsi(image, kernel, noise):
    """Apply LSI degradation: G = H * F + N via circular convolution (FFT)."""
    k_embed = embed_kernel(kernel, image.shape)
    degraded = np.real(ifft2(fft2(image) * fft2(k_embed)))
    return degraded + noise


def part_2_1():
    print("=" * 60)
    print("Part 2.1: LSI Degradation Model")
    print("=" * 60)

    img = load_image("trui.png")
    rng = np.random.default_rng(42)
    kernel = gaussian_kernel_2d(15, sigma=3)
    noise_configs = [(0, "No noise (σ=0)"), (10, "Low noise (σ=10)"), (40, "High noise (σ=40)")]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (trui.png)")
    axes[0].axis("off")

    for i, (sigma_n, label) in enumerate(noise_configs):
        noise = rng.normal(0, sigma_n, img.shape) if sigma_n > 0 else np.zeros_like(img)
        degraded = apply_lsi(img, kernel, noise)
        axes[i + 1].imshow(np.clip(degraded, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[i + 1].set_title(f"Gaussian blur (σ_k=3)\n{label}")
        axes[i + 1].axis("off")

    plt.suptitle("LSI Degradation Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_1.png"), dpi=150)
    plt.close()
    print("  Saved part_2_1.png")
    print()


# =============================================================================
# Part 2.2 – Direct Inverse Filtering
# =============================================================================

def direct_inverse_filter(degraded, kernel, epsilon=1e-3):
    """
    Direct inverse filter: F_hat = G / H.
    Uses epsilon threshold to avoid division by near-zero H values.
    """
    G = fft2(degraded)
    H = fft2(embed_kernel(kernel, degraded.shape))
    H_safe = np.where(np.abs(H) > epsilon, H, epsilon)
    return np.real(ifft2(G / H_safe))


def part_2_2():
    print("=" * 60)
    print("Part 2.2: Direct Inverse Filtering")
    print("=" * 60)

    img = load_image("trui.png")
    rng = np.random.default_rng(42)
    kernel = gaussian_kernel_2d(15, sigma=3)
    noise_levels = [(0, "σ_n=0"), (5, "σ_n=5"), (30, "σ_n=30")]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, (sigma_n, label) in enumerate(noise_levels):
        noise = rng.normal(0, sigma_n, img.shape) if sigma_n > 0 else np.zeros_like(img)
        degraded = apply_lsi(img, kernel, noise)
        restored = direct_inverse_filter(degraded, kernel)

        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(degraded, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(f"Degraded ({label})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(np.clip(restored, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[i, 2].set_title(f"Direct Inverse Filter ({label})")
        axes[i, 2].axis("off")

    plt.suptitle("Direct Inverse Filtering (ε=1e-3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_2.png"), dpi=150)
    plt.close()
    print("  Saved part_2_2.png")
    print()


# =============================================================================
# Part 2.3 – Wiener Filter
# =============================================================================

def wiener_filter(degraded, kernel, K):
    """
    Wiener filter (Gonzales & Woods eq. 5.8-6):
        W = H* / (|H|^2 + K)
    K approximates the noise-to-signal power spectral density ratio.
    """
    G = fft2(degraded)
    H = fft2(embed_kernel(kernel, degraded.shape))
    W = np.conj(H) / (np.abs(H)**2 + K)
    return np.real(ifft2(W * G))


def part_2_3():
    print("=" * 60)
    print("Part 2.3: Wiener Filter")
    print("=" * 60)

    img = load_image("trui.png")
    rng = np.random.default_rng(42)
    kernel = gaussian_kernel_2d(15, sigma=3)

    # (noise_sigma, K, label)
    configs = [
        (5,  0.001, "σ_n=5,  K=0.001"),
        (5,  0.1,   "σ_n=5,  K=0.1"),
        (30, 0.001, "σ_n=30, K=0.001"),
        (30, 0.1,   "σ_n=30, K=0.1"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for i, (sigma_n, K, label) in enumerate(configs):
        noise = rng.normal(0, sigma_n, img.shape)
        degraded = apply_lsi(img, kernel, noise)
        restored = wiener_filter(degraded, kernel, K)

        axes[0, i].imshow(np.clip(degraded, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"Degraded ({label})", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(restored, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"Wiener ({label})", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Wiener Filter (Gonzales & Woods eq. 5.8-6)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_3.png"), dpi=150)
    plt.close()
    print("  Saved part_2_3.png")
    print()


# =============================================================================
# Part 3.1 – Canny Edge Detection (hand.tiff)
# =============================================================================

def part_3_1():
    print("=" * 60)
    print("Part 3.1: Canny Edge Detection (hand.tiff)")
    print("=" * 60)

    img = load_image("hand.tiff")
    img_norm = img / img.max()  # skimage canny expects float in [0,1]

    configs = [
        (1.0, 0.05, 0.15, "σ=1.0  low=0.05 high=0.15\n(default-like)"),
        (3.0, 0.05, 0.15, "σ=3.0  low=0.05 high=0.15\n(smooth, fewer edges)"),
        (1.0, 0.02, 0.06, "σ=1.0  low=0.02 high=0.06\n(sensitive, more edges)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original (hand.tiff)")
    axes[0].axis("off")

    for i, (sigma, low, high, label) in enumerate(configs):
        edges = canny(img_norm, sigma=sigma, low_threshold=low, high_threshold=high)
        axes[i + 1].imshow(edges, cmap="gray")
        axes[i + 1].set_title(label, fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Canny Edge Detection – Effect of Parameters", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_3_1.png"), dpi=150)
    plt.close()
    print("  Saved part_3_1.png")
    print()


# =============================================================================
# Part 3.2 – Harris Corner Response (modelhouses.png)
# =============================================================================

def part_3_2():
    print("=" * 60)
    print("Part 3.2: Harris Corner Response (modelhouses.png)")
    print("=" * 60)

    img = load_image("modelhouses.png")
    img_norm = img / img.max()

    # method='k' variations
    configs_k = [
        (1.0, 0.05, "method=k\nσ=1, k=0.05"),
        (3.0, 0.05, "method=k\nσ=3, k=0.05"),
        (2.0, 0.15, "method=k\nσ=2, k=0.15"),
    ]
    # method='eps' variations (fixed sigma=2)
    configs_eps = [
        (2.0, 0.01, "method=eps\nσ=2, eps=0.01"),
        (2.0, 0.1,  "method=eps\nσ=2, eps=0.1"),
    ]

    n_cols = 1 + len(configs_k) + len(configs_eps)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (sigma, k, label) in enumerate(configs_k):
        response = corner_harris(img_norm, method="k", sigma=sigma, k=k)
        axes[i + 1].imshow(response, cmap="hot")
        axes[i + 1].set_title(label, fontsize=9)
        axes[i + 1].axis("off")

    offset = 1 + len(configs_k)
    for i, (sigma, eps, label) in enumerate(configs_eps):
        response = corner_harris(img_norm, method="eps", sigma=sigma, eps=eps)
        axes[offset + i].imshow(response, cmap="hot")
        axes[offset + i].set_title(label, fontsize=9)
        axes[offset + i].axis("off")

    plt.suptitle("Harris Corner Response Maps – Effect of Parameters", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_3_2.png"), dpi=150)
    plt.close()
    print("  Saved part_3_2.png")
    print()


# =============================================================================
# Part 3.3 – 250 Strongest Harris Corners
# =============================================================================

def find_harris_corners(image, num_peaks=250, sigma=2.0, k=0.05, min_distance=5):
    """Find Harris corners as local maxima of the corner response map."""
    response = corner_harris(image, method="k", sigma=sigma, k=k)
    coords = corner_peaks(response, num_peaks=num_peaks, min_distance=min_distance)
    return coords, response


def part_3_3():
    print("=" * 60)
    print("Part 3.3: 250 Strongest Harris Corners (modelhouses.png)")
    print("=" * 60)

    img = load_image("modelhouses.png")
    img_norm = img / img.max()

    sigma, k = 2.0, 0.05
    coords, _ = find_harris_corners(img_norm, num_peaks=250, sigma=sigma, k=k)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap="gray")
    ax.plot(coords[:, 1], coords[:, 0], "r.", markersize=4, label=f"Corners (n={len(coords)})")
    ax.set_title(
        f"250 Strongest Harris Corners on modelhouses.png\n"
        f"(method=k, σ={sigma}, k={k}, min_distance=5)"
    )
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_3_3.png"), dpi=150)
    plt.close()
    print(f"  Detected {len(coords)} corners. Saved part_3_3.png")
    print()


# =============================================================================
# Part 4.1 – Scale-Space: Verify G(σ) * G(τ) = G(√(σ²+τ²))
# =============================================================================

def part_4_1():
    print("=" * 60)
    print("Part 4.1: Scale-Space Blob Visualization")
    print("=" * 60)

    grid_size = 128  # Power of 2 for efficient FFT; large enough for σ=1,τ=2
    sigma, tau = 1.0, 2.0
    sigma_combined = np.sqrt(sigma**2 + tau**2)

    # FFT-convention coordinates: 0..N-1 where indices > N//2 map to negative coords
    idx = np.arange(grid_size)
    X_idx, Y_idx = np.meshgrid(idx, idx)
    X = np.where(X_idx <= grid_size // 2, X_idx.astype(float),
                 X_idx.astype(float) - grid_size)
    Y = np.where(Y_idx <= grid_size // 2, Y_idx.astype(float),
                 Y_idx.astype(float) - grid_size)

    B = gaussian_2d(X, Y, sigma)            # blob B = G(σ=1), peak at (0,0) corner
    G_tau = gaussian_2d(X, Y, tau)           # Gaussian G(τ=2), peak at (0,0) corner
    G_comb = gaussian_2d(X, Y, sigma_combined)  # Direct G(√5), peak at (0,0) corner

    # Convolve B with G(τ) via FFT (circular, effectively linear since Gaussians decay fast)
    result_conv = np.real(ifft2(fft2(B) * fft2(G_tau)))
    diff = result_conv - G_comb

    # Shift origin to centre for display
    B_vis     = fftshift(B)
    conv_vis  = fftshift(result_conv)
    comb_vis  = fftshift(G_comb)
    diff_vis  = fftshift(diff)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    data_titles = [
        (B_vis,    f"B = G(σ={sigma})\n(blob image)"),
        (conv_vis, f"B * G(τ={tau})\n[FFT convolution]"),
        (comb_vis, f"G(σ=√(σ²+τ²)={sigma_combined:.2f})\n[direct model]"),
        (diff_vis, "Difference\n(conv – direct)"),
    ]
    cmaps = ["viridis", "viridis", "viridis", "RdBu_r"]
    for ax, (data, title), cmap in zip(axes, data_titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Eq.(1) Verification: G(σ) * G(τ) = G(√(σ²+τ²))", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_4_1.png"), dpi=150)
    plt.close()
    print(f"  σ={sigma}, τ={tau}, σ_combined={sigma_combined:.4f}")
    print(f"  Max |difference|: {np.max(np.abs(diff)):.2e}")
    print("  Saved part_4_1.png")
    print()


# =============================================================================
# Part 4.3.iii – Plot H(0,0,τ) vs τ
# =============================================================================

def part_4_3_iii():
    print("=" * 60)
    print("Part 4.3.iii: H(0,0,τ) vs τ")
    print("=" * 60)

    # Analytical derivation:
    #   I(x,y,τ)  = B(x,y) * G(x,y,τ) = G(x,y,√(σ²+τ²))   [eq. 1, B=G(σ)]
    #   Let s² = σ² + τ²
    #   ∂²I/∂x²  = I · (x²/s⁴ - 1/s²)
    #   ∂²I/∂y²  = I · (y²/s⁴ - 1/s²)
    #   ∇²I      = I · ((x²+y²)/s⁴ - 2/s²)
    #   H(x,y,τ) = τ^(γ·2) · ∇²I  with γ=1  →  H = τ² · ∇²I
    #
    #   At (0,0):
    #     I(0,0,τ) = 1/(2πs²)
    #     ∇²I|_{0,0} = (1/(2πs²)) · (-2/s²)  = -1/(πs⁴)
    #     H(0,0,τ)   = τ² · (-1/(πs⁴))       = -τ² / (π(σ²+τ²)²)
    #
    #   Extremum: dH/dτ = 0  →  τ = σ  (minimum, since H(0,0,τ)<0 has maximum |value|)

    sigma = 1.0
    tau_vals = np.linspace(0.05, 10, 1000)
    s_sq = sigma**2 + tau_vals**2
    H00 = -tau_vals**2 / (np.pi * s_sq**2)

    tau_ext = sigma  # extremal τ
    H_ext = -(tau_ext**2) / (np.pi * (sigma**2 + tau_ext**2)**2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tau_vals, H00, "b-", linewidth=2, label=r"$H(0,0,\tau) = -\tau^2 / [\pi(\sigma^2+\tau^2)^2]$")
    ax.axvline(x=tau_ext, color="r", linestyle="--",
               label=f"Extremum at τ = σ = {tau_ext} (|H| is maximum)")
    ax.axhline(y=0, color="k", linewidth=0.7)
    ax.scatter([tau_ext], [H_ext], color="r", zorder=5)
    ax.set_xlabel(r"$\tau$ (scale parameter)", fontsize=12)
    ax.set_ylabel(r"$H(0,0,\tau)$", fontsize=12)
    ax.set_title(
        r"Scale-Normalized Laplacian at origin: $H(0,0,\tau) = -\tau^2/[\pi(\sigma^2+\tau^2)^2]$"
        f"\n(σ = {sigma})",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_4_3_iii.png"), dpi=150)
    plt.close()
    print(f"  H(0,0,τ) = -τ²/[π(σ²+τ²)²]  with σ={sigma}")
    print(f"  Extremal τ = σ = {tau_ext},  H(0,0,σ) = {H_ext:.6f}")
    print("  Saved part_4_3_iii.png")
    print()


# =============================================================================
# Part 4.4 – Blob Detection on sunflower.tiff (150 blobs)
# =============================================================================

def part_4_4():
    print("=" * 60)
    print("Part 4.4: Blob Detection on sunflower.tiff")
    print("=" * 60)

    # Load and optionally downsample for speed
    img_pil = Image.open(os.path.join(BASE, "sunflower.tiff")).convert("L")
    max_dim = 600
    if max(img_pil.size) > max_dim:
        scale = max_dim / max(img_pil.size)
        new_size = (int(img_pil.size[0] * scale), int(img_pil.size[1] * scale))
        img_pil = img_pil.resize(new_size, Image.LANCZOS)
    img = np.array(img_pil, dtype=np.float64)
    img_norm = (img - img.min()) / (img.max() - img.min())
    print(f"  Image size: {img.shape}")

    # Build scale-space of scale-normalized LoG responses
    # H(x,y,τ) = τ² · ∇²[G(τ) * I(x,y)] = τ² · gaussian_laplace(I, σ=τ)
    tau_vals = np.logspace(np.log10(2), np.log10(25), 20)
    H_stack = np.zeros((len(tau_vals), img.shape[0], img.shape[1]))
    for i, tau in enumerate(tau_vals):
        H_stack[i] = tau**2 * gaussian_laplace(img_norm, sigma=tau)

    print(f"  Scale range: τ ∈ [{tau_vals[0]:.1f}, {tau_vals[-1]:.1f}], {len(tau_vals)} scales")

    # Find local maxima (positive LoG: bright blobs on dark background)
    coords_max = peak_local_max(H_stack,  min_distance=5, threshold_abs=0.0)
    # Find local minima (negative LoG: dark blobs on bright background)
    coords_min = peak_local_max(-H_stack, min_distance=5, threshold_abs=0.0)

    # Gather all extrema with their H values
    all_coords = np.vstack([coords_max, coords_min])
    vals_max = H_stack[coords_max[:, 0], coords_max[:, 1], coords_max[:, 2]]
    vals_min = H_stack[coords_min[:, 0], coords_min[:, 1], coords_min[:, 2]]
    all_vals = np.concatenate([vals_max, vals_min])

    # Keep top 150 by |H|
    top_idx = np.argsort(-np.abs(all_vals))[:150]
    top_coords = all_coords[top_idx]  # shape (150, 3): (tau_idx, y, x)
    top_vals   = all_vals[top_idx]

    is_max = top_vals > 0
    print(f"  Top-150 extrema: {is_max.sum()} maxima (red), {(~is_max).sum()} minima (blue)")

    # Draw circles proportional to τ
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, cmap="gray")

    for (tau_idx, y, x), val in zip(top_coords[is_max], top_vals[is_max]):
        r = tau_vals[tau_idx] * np.sqrt(2)
        ax.add_patch(plt.Circle((x, y), r, color="red",  fill=False, linewidth=1.2))
        ax.plot(x, y, "r.", markersize=3)

    for (tau_idx, y, x), val in zip(top_coords[~is_max], top_vals[~is_max]):
        r = tau_vals[tau_idx] * np.sqrt(2)
        ax.add_patch(plt.Circle((x, y), r, color="cyan", fill=False, linewidth=1.2))
        ax.plot(x, y, "c.", markersize=3)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="red",  markerfacecolor="none",
               linestyle="none", label=f"Maxima – bright blobs (n={is_max.sum()})"),
        Line2D([0], [0], marker="o", color="cyan", markerfacecolor="none",
               linestyle="none", label=f"Minima – dark blobs (n={(~is_max).sum()})"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=10)
    ax.set_title(
        f"Scale-Space Blob Detection: top-150 LoG extrema on sunflower.tiff\n"
        f"τ ∈ [{tau_vals[0]:.1f}, {tau_vals[-1]:.1f}], 20 scales, radius ∝ τ√2",
        fontsize=11
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_4_4.png"), dpi=150)
    plt.close()
    print("  Saved part_4_4.png")
    print()


# =============================================================================
# Main – Run all parts
# =============================================================================

if __name__ == "__main__":
    part_1_1()
    part_2_1()
    part_2_2()
    part_2_3()
    part_3_1()
    part_3_2()
    part_3_3()
    part_4_1()
    part_4_3_iii()
    part_4_4()
    print("=" * 60)
    print("All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)
