import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.color import rgb2hsv, hsv2rgb


##### 1.1 #####
def gamma_transform(I, gamma):
    return I ** gamma

I = imread("tiger.jpg")
I = img_as_float(I)

if I.ndim == 3:
    I = rgb2gray(I)

gammas = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, len(gammas)+1, figsize=(14,4))
axes[0].imshow(I, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

# Apply gamma correction to image for each value of gamma
for i, g in enumerate(gammas):
    J = gamma_transform(I, g)
    axes[i+1].imshow(J, cmap="gray")
    axes[i+1].set_title(f"gamma = {g}")
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()

##### 1.2 #####

I = imread("autumn.tif")
I = img_as_float(I)

gamma = 0.5

# Apply gamma correction to each RGB channel separately
R = gamma_transform(I[:, :, 0], gamma)
G = gamma_transform(I[:, :, 1], gamma)
B = gamma_transform(I[:, :, 2], gamma)

# Recombine channels
I_gamma = np.stack((R, G, B), axis=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(I)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(I_gamma)
axes[1].set_title(f"Gamma corrected (gamma = {gamma})")
axes[1].axis("off")

plt.tight_layout()
plt.show()

##### 1.3 #####

I_rgb = img_as_float(imread("autumn.tif"))
# Convert to HSV
I_hsv = rgb2hsv(I_rgb)

# Apply gamma correction to V channel only
I_hsv_gamma = I_hsv.copy()
I_hsv_gamma[:, :, 2] = gamma_transform(I_hsv_gamma[:, :, 2], gamma)  # V channel

# Convert back to RGB
I_rgb_gamma = hsv2rgb(I_hsv_gamma)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(I_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(I_rgb_gamma)
axes[1].set_title(f"HSV V-channel gamma (gamma = {gamma})")
axes[1].axis("off")

plt.tight_layout()
plt.show()