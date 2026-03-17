import numpy as np
from skimage import io, color, feature
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.feature import canny, corner_harris, corner_peaks, peak_local_max

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
    img1_edges = feature.canny(img1_greyscale, sigma= 1.0 ) # apply canny edge detection
    save_image(img1_edges, "matrikelnumre_art edges", output_folder="output", filename=None, cmap="gray")



################## Part 2 #################################################


def find_harris_corners(image, num_peaks=250, sigma=2.0, k=0.05, min_distance=5):
    """Find Harris corners as local maxima of the corner response map."""
    response = corner_harris(image, method="k", sigma=sigma, k=k)
    coords = corner_peaks(response, num_peaks=num_peaks, min_distance=min_distance)
    return coords, response


if __name__ == "__main__":

    part_1_1()
    part_1_2()