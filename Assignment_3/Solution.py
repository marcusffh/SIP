"""
Assignment 3 – Fourier Transform (Part 2: In Practice)
Signal and Image Processing
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – save figures without display

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE = os.path.dirname(__file__)


def load_image(filename):
    """Load a grayscale image as a float64 numpy array."""
    path = os.path.join(BASE, filename)
    return np.array(Image.open(path), dtype=np.float64)


def power_spectrum(image):
    """Compute the centered power spectrum of a 2D image."""
    F = fft2(image)
    F_shifted = fftshift(F)
    return np.abs(F_shifted) ** 2


# =============================================================================
# Part 2.1 – Power Spectrum of trui.png
# =============================================================================
def part_2_1():
    print("=" * 60)
    print("Part 2.1: Power Spectrum of trui.png")
    print("=" * 60)

    img = load_image("trui.png")
    ps = power_spectrum(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image (trui.png)")
    axes[0].axis("off")

    # Display log-scaled power spectrum for visibility
    im = axes[1].imshow(np.log1p(ps), cmap="gray")
    axes[1].set_title("Log Power Spectrum (after fftshift)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_1.png"), dpi=150)
    plt.show()

    print(
        "Interpretation: fftshift moves the zero-frequency component to the "
        "center of the spectrum. The bright spot at the center represents the "
        "DC component (average intensity). The spread of energy around the "
        "center reflects the spatial frequency content of the image – edges "
        "and fine details contribute to higher frequencies (farther from center)."
    )
    print()


# =============================================================================
# Part 2.2 – Cosine Wave Addition and Filtering on cameraman.tif
# =============================================================================
def part_2_2():
    print("=" * 60)
    print("Part 2.2: Cosine Wave Addition & Filtering (cameraman.tif)")
    print("=" * 60)

    img = load_image("cameraman.tif")
    M, N = img.shape

    # Parameters for the cosine wave
    a0 = 100.0
    v0 = 30.0   # cycles across image width
    w0 = 50.0   # cycles across image height

    # Create coordinate grids (pixel indices)
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)

    # Add planar cosine wave: a0 * cos(2*pi*(v0*x/N + w0*y/M))
    cosine_wave = a0 * np.cos(2 * np.pi * (v0 * X / N + w0 * Y / M))
    img_noisy = img + cosine_wave

    # Compute power spectra
    ps_noisy = power_spectrum(img_noisy)

    # Design a notch filter to remove the cosine peaks
    # The cosine cos(2*pi*(v0*x/N + w0*y/M)) produces two impulses in the
    # frequency domain at (v0, w0) and (-v0, -w0), which after fftshift
    # appear at (M//2 + w0, N//2 + v0) and (M//2 - w0, N//2 - v0).
    F_noisy = fftshift(fft2(img_noisy))

    # Create notch filter (reject at the two impulse locations)
    notch_filter = np.ones((M, N), dtype=np.float64)
    notch_radius = 5  # radius around the peaks to suppress

    cy, cx = M // 2, N // 2
    # Peak locations in the shifted spectrum
    peak1 = (cy + int(w0), cx + int(v0))
    peak2 = (cy - int(w0), cx - int(v0))

    YY, XX = np.ogrid[:M, :N]
    for py, px in [peak1, peak2]:
        dist = np.sqrt((YY - py) ** 2 + (XX - px) ** 2)
        notch_filter[dist <= notch_radius] = 0.0

    # Apply filter and reconstruct
    F_filtered = F_noisy * notch_filter
    img_filtered = np.real(ifft2(ifftshift(F_filtered)))

    ps_filtered = np.abs(F_filtered) ** 2

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_noisy, cmap="gray")
    axes[0, 1].set_title(f"With cosine (v0={v0}, w0={w0})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.log1p(ps_noisy), cmap="gray")
    axes[0, 2].set_title("Power Spectrum (noisy)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(notch_filter, cmap="gray")
    axes[1, 0].set_title("Notch Filter")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.log1p(ps_filtered), cmap="gray")
    axes[1, 1].set_title("Power Spectrum (filtered)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_filtered, cmap="gray")
    axes[1, 2].set_title("Filtered Image")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_2.png"), dpi=150)
    plt.show()

    print(
        f"A cosine wave a0*cos(2*pi*(v0*x/N + w0*y/M)) with a0={a0}, v0={v0}, "
        f"w0={w0} was added to the cameraman image. In the power spectrum, this "
        "creates two bright impulse peaks at (+v0,+w0) and (-v0,-w0) relative "
        "to the center. A notch filter (radius=5 pixels) centered on these peaks "
        "suppresses them, effectively removing the cosine wave from the image."
    )
    print()


# =============================================================================
# Part 2.3 – Radial Average of Power Spectrum
# =============================================================================
def radial_average_power_spectrum(image):
    """
    Compute the average power spectrum at each spatial frequency,
    where spatial frequency = Euclidean distance from the center.

    Returns:
        frequencies: array of integer distances from the center
        avg_power: average power at each distance
    """
    ps = power_spectrum(image)
    M, N = ps.shape
    cy, cx = M // 2, N // 2

    # Compute distance of each pixel from the center
    y, x = np.ogrid[:M, :N]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    # Maximum distance from center to edge (not corner)
    max_dist = min(cy, cx)

    # Bin by integer distance
    dist_int = np.round(dist).astype(int)
    frequencies = np.arange(0, max_dist + 1)
    avg_power = np.zeros(len(frequencies))

    for r in frequencies:
        mask = dist_int == r
        if np.any(mask):
            avg_power[r] = np.mean(ps[mask])

    return frequencies, avg_power


def part_2_3():
    print("=" * 60)
    print("Part 2.3: Radial Average Power Spectrum")
    print("=" * 60)

    img_bigben = load_image("bigben_cropped_gray.png")

    # Create random noise image with same dimensions
    rng = np.random.default_rng(42)
    img_noise = rng.standard_normal(img_bigben.shape) * 128 + 128

    freq_bb, avg_bb = radial_average_power_spectrum(img_bigben)
    freq_noise, avg_noise = radial_average_power_spectrum(img_noise)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq_bb[1:], avg_bb[1:], label="Big Ben", linewidth=1.5)
    ax.loglog(freq_noise[1:], avg_noise[1:], label="Random Noise", linewidth=1.5)
    ax.set_xlabel("Spatial Frequency (distance from center)")
    ax.set_ylabel("Average Power")
    ax.set_title("Radial Average Power Spectrum (log-log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_3.png"), dpi=150)
    plt.show()

    print(
        "The Big Ben image shows a steep decline in average power with "
        "increasing spatial frequency (approximately following a power law), "
        "which is characteristic of natural images – most energy is concentrated "
        "in low frequencies (smooth regions), with progressively less energy at "
        "higher frequencies (fine details/edges). The random noise image has a "
        "relatively flat power spectrum across all frequencies, as expected for "
        "white noise where all frequencies contribute equally."
    )
    print()


# =============================================================================
# Part 2.4 – Angular Average of Power Spectrum
# =============================================================================
def angular_average_power_spectrum(image, freq_range=(10, 100), angle_bin_size=10):
    """
    Compute the average power spectrum at each angle within a specified
    range of spatial frequencies.

    Args:
        image: 2D grayscale image
        freq_range: (min_freq, max_freq) range of spatial frequencies to consider
        angle_bin_size: size of angular bins in degrees

    Returns:
        angles: center of each angular bin in degrees [0, 360)
        avg_power: average power in each angular bin
    """
    ps = power_spectrum(image)
    M, N = ps.shape
    cy, cx = M // 2, N // 2

    y, x = np.ogrid[:M, :N]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    # Angle from center (0 to 360 degrees)
    angle = np.degrees(np.arctan2(y - cy, x - cx)) % 360

    # Mask for the frequency range
    freq_mask = (dist >= freq_range[0]) & (dist <= freq_range[1])

    # Create angular bins
    n_bins = int(360 / angle_bin_size)
    angles = np.arange(0, 360, angle_bin_size)
    avg_power = np.zeros(n_bins)

    for i, a in enumerate(angles):
        a_low = a
        a_high = a + angle_bin_size
        angle_mask = (angle >= a_low) & (angle < a_high)
        combined_mask = freq_mask & angle_mask
        if np.any(combined_mask):
            avg_power[i] = np.mean(ps[combined_mask])

    return angles, avg_power


def plot_angular_template(freq_range=(10, 100), angle_bin_size=10, image_shape=(480, 480)):
    """
    Re-implement the angular bin template shown in Figure 1 of the assignment.
    """
    M, N = image_shape
    cy, cx = M // 2, N // 2
    size = max(M, N)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    n_bins = int(360 / angle_bin_size)
    angles_rad = np.deg2rad(np.arange(0, 360, angle_bin_size))

    # Draw angular bins as sectors in the frequency range
    for a_rad in angles_rad:
        width = np.deg2rad(angle_bin_size)
        # Normalize radii for display
        r_inner = freq_range[0] / (size // 2)
        r_outer = freq_range[1] / (size // 2)
        ax.bar(
            a_rad, r_outer - r_inner, width=width, bottom=r_inner,
            edgecolor="black", linewidth=0.5, fill=True, alpha=0.3, color="blue"
        )

    ax.set_title(
        f"Angular Bins ({angle_bin_size}\u00b0 bins, "
        f"freq range [{freq_range[0]}-{freq_range[1]}])",
        pad=20
    )
    ax.set_thetagrids(np.arange(0, 360, angle_bin_size))
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_4_template.png"), dpi=150)
    plt.show()


def part_2_4():
    print("=" * 60)
    print("Part 2.4: Angular Average Power Spectrum")
    print("=" * 60)

    img_bigben = load_image("bigben_cropped_gray.png")

    rng = np.random.default_rng(42)
    img_noise = rng.standard_normal(img_bigben.shape) * 128 + 128

    freq_range = (10, 100)
    angle_bin_size = 10

    # Plot the template
    plot_angular_template(freq_range, angle_bin_size, img_bigben.shape)

    # Compute angular averages
    angles_bb, avg_bb = angular_average_power_spectrum(
        img_bigben, freq_range, angle_bin_size
    )
    angles_noise, avg_noise = angular_average_power_spectrum(
        img_noise, freq_range, angle_bin_size
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = angle_bin_size * 0.4
    ax.bar(
        angles_bb - bar_width / 2, avg_bb, width=bar_width,
        label="Big Ben", alpha=0.8
    )
    ax.bar(
        angles_noise + bar_width / 2, avg_noise, width=bar_width,
        label="Random Noise", alpha=0.8
    )
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Average Power")
    ax.set_title("Angular Average Power Spectrum")
    ax.set_xticks(np.arange(0, 360, angle_bin_size))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_4.png"), dpi=150)
    plt.show()

    print(
        "The Big Ben image shows peaks in the angular power distribution at "
        "around 0/180 degrees (horizontal) and 90/270 degrees (vertical). This "
        "reflects the strong vertical and horizontal structures in the building "
        "(walls, edges, columns). The random noise image has a roughly uniform "
        "angular distribution, since noise has no preferred orientation."
    )
    print()


# =============================================================================
# Part 2.5 – Spatial Derivatives via Frequency Domain
# =============================================================================
def frequency_domain_derivative(image, dx_order=0, dy_order=0):
    """
    Compute partial derivatives of a 2D image using the frequency domain.

    By the convolution theorem, differentiation in the spatial domain
    corresponds to multiplication by (j*2*pi*k)^n in the frequency domain,
    where k is the frequency variable and n is the derivative order.

    Args:
        image: 2D grayscale image (numpy array)
        dx_order: order of derivative in x direction
        dy_order: order of derivative in y direction

    Returns:
        derivative: the partial derivative of the image (real-valued)
    """
    M, N = image.shape
    F = fft2(image)

    # Frequency coordinates (not shifted – matching fft2 output convention)
    kx = fftfreq(N, d=1.0)  # cycles per pixel
    ky = fftfreq(M, d=1.0)

    KX, KY = np.meshgrid(kx, ky)

    # Derivative kernel in frequency domain: (j*2*pi*k)^order
    H = (1j * 2 * np.pi * KX) ** dx_order * (1j * 2 * np.pi * KY) ** dy_order

    # Multiply and inverse transform
    result = np.real(ifft2(F * H))
    return result


def part_2_5():
    print("=" * 60)
    print("Part 2.5: Spatial Derivatives via Frequency Domain")
    print("=" * 60)

    img = load_image("cameraman.tif")

    # Compute derivatives at different orders
    derivatives = [
        (1, 0, r"$\partial f / \partial x$"),
        (0, 1, r"$\partial f / \partial y$"),
        (2, 0, r"$\partial^2 f / \partial x^2$"),
        (0, 2, r"$\partial^2 f / \partial y^2$"),
        (1, 1, r"$\partial^2 f / \partial x \partial y$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Show original
    im0 = axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original (cameraman.tif)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    for i, (dx, dy, title) in enumerate(derivatives):
        deriv = frequency_domain_derivative(img, dx_order=dx, dy_order=dy)
        im = axes[i + 1].imshow(deriv, cmap="RdBu_r")
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part_2_5.png"), dpi=150)
    plt.show()

    print(
        "Spatial derivatives are computed by multiplying the Fourier transform "
        "of the image with (j*2*pi*kx)^dx_order * (j*2*pi*ky)^dy_order, then "
        "taking the inverse FFT. First-order derivatives highlight edges in "
        "the respective direction (x or y). Second-order derivatives emphasize "
        "fine details and are sensitive to rapid intensity changes. The mixed "
        "derivative highlights diagonal structures."
    )
    print()


# =============================================================================
# Main – Run all parts
# =============================================================================
if __name__ == "__main__":
    part_2_1()
    part_2_2()
    part_2_3()
    part_2_4()
    part_2_5()
    print("All outputs saved to:", OUTPUT_DIR)
