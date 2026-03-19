import numpy as np
from skimage import io, color, feature
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter
from skimage.filters import gaussian
from skimage.feature import canny, corner_harris, corner_peaks, peak_local_max
from skimage.transform import ProjectiveTransform, warp, hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage.morphology import binary_opening, binary_closing, disk


#Load image in greyscale, and convert to numpy array
def load_image_greyscale(filename):
    img = Image.open(filename).convert("L")
    return np.array(img, dtype=np.uint8)


def save_image(image, title, output_folder="output", filename=None, cmap="gray"):
    """
    Saves an image with a title to a folder without displaying it.

    Parameters:
    - image: numpy array (grayscale or RGB)
    - title: string (used as figure title + filename fallback)
    - output_folder: string
    - filename: optional custom filename
    - cmap: colormap for grayscale images
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Auto-generate filename if not provided
    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"

    save_path = os.path.join(output_folder, filename)

    # Create figure WITHOUT showing
    fig = plt.figure()
    
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

    plt.title(title)
    plt.axis("off")

    # Save and immediately close (prevents display)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved to: {save_path}")

def save_image_with_corners(image, coords, title, output_folder="output", filename=None, cmap="gray"):
    """
    Saves an image with overlaid corner points.

    Parameters:
    - image: numpy array (grayscale or RGB)
    - coords: array of shape (N, 2) with (row, col) positions
    - title: string
    - output_folder: string
    - filename: optional filename
    - cmap: colormap for grayscale images
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Auto filename
    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"

    save_path = os.path.join(output_folder, filename)

    # Create figure
    fig = plt.figure()

    # Show image
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

    # Plot corner points
    if coords is not None and len(coords) > 0:
        plt.plot(coords[:, 1], coords[:, 0], "r.", markersize=4)

    plt.title(title)
    plt.axis("off")

    # Save and close
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved to: {save_path}")

def save_intensity_histogram(image, bins,title,  output_folder="output", filename= None):
    """
    Saves a pixel intensity histogram to disk instead of displaying it.

    Parameters:
    - image: numpy array (assumes values in [0,255])
    - bins: number of bins
    - output_folder: directory to save the image
    - filename: output file name
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Auto-generate filename if not provided
    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"

    save_path = os.path.join(output_folder, filename)

    # Flatten image
    values = image.flatten()

    # Create figure
    fig = plt.figure()

    plt.hist(values, bins=bins, range=(0, 255), color='gray', alpha=1)
    plt.yscale('linear')
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Number of Pixels")

    # Save and close
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Histogram saved to: {save_path}")

############ Part 1 ######################################
############otsu segmentation############

def otsu_threshold_applied(image): #assumes greyscale image in intensity range (0-255)
    img_array = np.array(image).astype(np.uint8) # convert to numpy array
    threshold = threshold_otsu(img_array) #find threshold using otsu
    thresholded_img = np.where(img_array >=threshold, 255,0) #threshold
    return  thresholded_img

def intensity_histogram(image, bins): # assumes int in range (0-255) and greyscale image
    values = image.flatten()
    plt.hist(values, bins=bins, range=(0, 255), color='gray', alpha=1)
    plt.yscale('log')
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Number of Pixels")
    plt.show()


def part_1_1():
    #load image
    img1_greyscale = load_image_greyscale("input/matrikelnumre_art.png")# greyscale
    img2_greyscale = load_image_greyscale("input/matrikelnumre_nat.png")# greyscale

    #perform threshold segmentation
    img1_segmented = otsu_threshold_applied(img1_greyscale)
    img2_segmented = otsu_threshold_applied(img2_greyscale)

    # output greyscale and segmented images
    save_image(img1_segmented, "matrikelnumre_art segmented", output_folder="output", filename=None, cmap="gray")
    save_image(img2_segmented, "matrikelnumre_nat segmented", output_folder="output", filename=None, cmap="gray")
    save_image(img1_greyscale, "matrikelnumre_art", output_folder="output", filename=None, cmap="gray")
    save_image(img2_greyscale, "matrikelnumre_nat", output_folder="output", filename=None, cmap="gray")

    #output histograms of greyscale images
    save_intensity_histogram(img1_greyscale, bins = 256, title= "matrikelnumre_art intensity-histogram",  output_folder="output", filename= None)
    save_intensity_histogram(img2_greyscale, bins = 256, title= "matrikelnumre_nat intensity-histogram",  output_folder="output", filename= None)

############################## Canny edge detection

def part_1_2():
    img1_greyscale = load_image_greyscale("input/matrikelnumre_art.png")# load in greyscale
    img1_edges = feature.canny(img1_greyscale, sigma= 1.0) # apply canny edge detection
    save_image(img1_edges, "matrikelnumre_art edges", output_folder="output", filename=None, cmap="gray")



################## Part 2 #################################################


def find_harris_corners(image, num_peaks=250, sigma=2.0, k=0.05, min_distance=5):
    """Find Harris corners as local maxima of the corner response map."""
    response = corner_harris(image, method="k", sigma=sigma, k=k)
    coords = corner_peaks(response, num_peaks=num_peaks, min_distance=min_distance)
    return coords, response

# maybe

def part_2_1():

    # load image
    img1_greyscale = load_image_greyscale("input/matrikelnumre_nat.png")# greyscale
    img1 = np.array(Image.open("input/matrikelnumre_nat.png"))

    #preprocess
    img1_blur = gaussian(image = img1_greyscale, sigma = 9)
    img1_blur = (img1_blur * 255).astype(np.uint8)
    img1_segmented = otsu_threshold_applied(img1_blur)


    # find corners
    coords, response = find_harris_corners(image = img1_segmented, num_peaks= 8, sigma= 8.0, k=0.05, min_distance= 10)
    print(f"Corner cordinates are {coords}")

    #save image
    save_image_with_corners(img1, coords, "matrikelnumre_nat corners" , output_folder="output", filename=None)
    save_image(img1_blur,  "matrikelnumre_nat gaussian blur sigma = 8", output_folder="output", filename=None, cmap="gray")
    save_image(img1_segmented,  "matrikelnumre_nat otsu segmentation", output_folder="output", filename=None, cmap="gray")



###### ########################################



def birds_eye_view(image, corners):
    """
    Bird's-eye transform using manual corners.

    corners_xy: shape (4,2) in order:
        [top-left, top-right, bottom-left, bottom-right]
        each corner = (x, y) = (col, row)
    """
    h, w = image.shape[:2] # [:2] for handling color images

    # Destination rectangle
    src = np.array([
        [0, 0],        # top-left
        [w-1, 0],      # top-right
        [0, h-1],      # bottom-left
        [w-1, h-1],    # bottom-right
    ])

    tform = ProjectiveTransform()
    tform.estimate(src, corners)

    warped = warp(image, tform, output_shape=(h, w))

    return warped


def part_2_2():
            # load image
    img1 = np.array(Image.open("input/matrikelnumre_nat.png"))

    corners = [[302,387],[170, 1355], [1010,410], [779,1606]] # manually typed from printed output from task 2.1
        # Swap to (x, y) from row colum
    corners = np.array([[c[1], c[0]] for c in corners])

    #[top-left, top-right, bottom-left, bottom-right]
    img1_transformed = birds_eye_view(img1, corners)

    save_image(img1_transformed,  "matrikelnumre_nat transformed", output_folder="output", filename=None)


########## Part 3 ############


def implemented_hough_line(edge_image, num_thetas=180):
    """
    Compute the Hough transform for straight lines

    Parameters:
    edge_image : 2D ndarray, binary edge image where nonzero pixels are treated as edge points
    num_thetas : int, the number of theta samples in [0, pi)

    Returns:
    H : 2D ndarray, Hough accumulator array
    rhos : 1D ndarray, discretized rho values
    thetas : 1D ndarray, discretized theta values in radians
    """
    height, width = edge_image.shape

    # maximum possible distance from origin = image diagonal
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))

    # discretize rho and theta
    rhos = np.arange(-diag_len, diag_len + 1)
    thetas = np.linspace(0, np.pi, num_thetas, endpoint=False)

    # accumulator
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # precompute sin and cos
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # find edge pixels
    ys, xs = np.nonzero(edge_image)

    # vote in accumulator
    for x, y in zip(xs, ys):
        for theta_idx in range(len(thetas)):
            rho = x * cos_t[theta_idx] + y * sin_t[theta_idx]
            rho_idx = int(np.round(rho)) + diag_len
            H[rho_idx, theta_idx] += 1

    return H, rhos, thetas


def find_hough_peaks(H, num_peaks=4, threshold=0.5, neighborhood_size=9):
    H_max = maximum_filter(H, size=neighborhood_size)
    peaks_mask = (H == H_max) & (H > threshold * H.max())

    peak_indices = np.argwhere(peaks_mask)
    peak_values = [H[r, t] for r, t in peak_indices]
    sorted_idx = np.argsort(peak_values)[::-1]

    peak_indices = peak_indices[sorted_idx]
    return peak_indices[:num_peaks]

def draw_lines(ax, image, peaks, rhos, thetas, title, color='r'):
    ax.imshow(image, cmap='gray')
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))

    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = x0 + 1000 * (-b)
        y1 = y0 + 1000 * (a)
        x2 = x0 - 1000 * (-b)
        y2 = y0 - 1000 * (a)

        ax.plot((x1, x2), (y1, y2), color=color, linewidth=2)

    ax.set_title(title)
    ax.axis('off')

img = io.imread("input/cross.png")


if img.ndim == 3:
    gray = color.rgb2gray(img)
else:
    gray = img

edges = gray > 0.5

# implemented Hough transform
H_my, rhos_my, thetas_my = implemented_hough_line(edges, num_thetas=180)
peaks_my = find_hough_peaks(H_my, num_peaks=4, threshold=0.4, neighborhood_size=9)

# skimage Hough transform
H_sk, thetas_sk, rhos_sk = hough_line(edges)
accums, angles_sk, dists_sk = hough_line_peaks(H_sk, thetas_sk, rhos_sk, num_peaks=4)

# convert skimage peaks to index pairs for reuse in drawing
peaks_sk = []
for angle, dist in zip(angles_sk, dists_sk):
    theta_idx = np.argmin(np.abs(thetas_sk - angle))
    rho_idx = np.argmin(np.abs(rhos_sk - dist))
    peaks_sk.append((rho_idx, theta_idx))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(edges, cmap='gray')
axes[0, 0].set_title("Input image: cross.png")
axes[0, 0].axis('off')

axes[0, 1].imshow(
    H_my,
    cmap='hot',
    aspect='auto',
    extent=[np.rad2deg(thetas_my[0]), np.rad2deg(thetas_my[-1]), rhos_my[-1], rhos_my[0]]
)
axes[0, 1].set_title("Implemented Hough transform")
axes[0, 1].set_xlabel("Theta (degrees)")
axes[0, 1].set_ylabel("Rho")

draw_lines(axes[1, 0], edges, peaks_my, rhos_my, thetas_my, "Implemented detected lines", color='red')
draw_lines(axes[1, 1], edges, peaks_sk, rhos_sk, thetas_sk, "skimage detected lines", color='lime')

plt.tight_layout()
plt.show()



def detect_blue_circle(img_transformed):
    """
    Detect the large blue circle in the transformed map image

    Parameters:
    img_transformed : ndarray, transformed bird's-eye-view image

    Returns:
    cx, cy, radius : int, detected circle center coordinates and radius
    blue_mask : ndarray, binary mask of blue pixels
    edges : ndarray, edge image used for Hough circle detection
    """
    img = img_transformed[..., :3]

    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # Blue mask
    blue_mask = (
        (B > 0.5) &
        (B > R + 0.1) &
        (B > G + 0.1)
    )

    # Clean mask
    blue_mask = binary_opening(blue_mask, disk(2))
    blue_mask = binary_closing(blue_mask, disk(3))

    # Restrict search region
    h, w = blue_mask.shape
    y0, y1 = int(0.2 * h), int(0.6 * h)
    x0, x1 = int(0.3 * w), int(0.8 * w)

    roi = blue_mask[y0:y1, x0:x1]

    # Edge detection on ROI
    edges = canny(roi.astype(float), sigma=1.5)

    # Radius range for Hough circle
    hough_radii = np.arange(20, 60, 2)

    # Circle Hough transform
    hough_res = hough_circle(edges, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        total_num_peaks=1
    )

    if len(cx) == 0:
        raise ValueError("No circle detected")

    # Convert back to full image coordinates
    cx = cx[0] + x0
    cy = cy[0] + y0
    radius = radii[0]

    return cx, cy, radius, blue_mask, edges


def plot_result(img_transformed, cx, cy, radius):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_transformed)

    # Red center dot
    ax.plot(cx, cy, 'ro', markersize=6)

    # Red circle
    circle = plt.Circle((cx, cy), radius, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.set_title("Detected blue circle centre")
    ax.axis("off")
    plt.show()

# Load transformed image
img_transformed = np.array(Image.open("output/matrikelnumre_nat_transformed.png"))

# Detect circle
cx, cy, radius, blue_mask, edges = detect_blue_circle(img_transformed)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_transformed)
axes[0].set_title("Transformed image")
axes[0].axis("off")

axes[1].imshow(blue_mask, cmap="gray")
axes[1].set_title("Blue mask")
axes[1].axis("off")

axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Edges for Hough circle")
axes[2].axis("off")

plt.tight_layout()
plt.show()

plot_result(img_transformed, cx, cy, radius)

print(f"Circle centre (x, y): ({cx}, {cy})")
print(f"Radius: {radius}")



if __name__ == "__main__":

    #part_1_1()
    #part_1_2()
    #part_2_1()
    part_2_2()
