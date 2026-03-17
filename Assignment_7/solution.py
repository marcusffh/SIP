import numpy as np

############ Part 1 ######################################

# Otsu function from VIP problem 5
#Based on the 1979_otsu_IEEESys.pdf document and the slides segmentation.pdf from The VIP course
def otsu(image):

    # Compute histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))

    # Normalize historgram and regard it as a probability distribution P(i) = h(i) /N
    p_i = histogram / np.sum(histogram) 

    total_mean = np.sum(np.arange(256) * p_i)  # mean intensity of the image

    max_variance = 0 # initial maximum variance
    threshold = 0   # initial threshold
    w0 = 0  # initial  probability for class 0
    w1 = 0  # initial probability for class 1 
    mu_k = 0.0
    epsilon = 1e-10
    L = len(p_i)  # number of bins = 256

    for i in range(L):
        w0 += p_i[i]  #  probability for class 0 cumulative (equation 2)
        w1 = 1 - w0  #  probability for class 1 (equation 3)

        if w0 < epsilon or w1 < epsilon: # makes sure we dont divide by zero:)
            continue

        mu_k += i * p_i[i]  # cumulative mean up to bin i

        mu0 =  mu_k / w0  # mean for class 0 (equation 4)

        mu1 = (total_mean - mu_k) / w1  # mean for class 1 (equation 5) 

        # Between class variance
        variance = w0 * w1 * (mu1 - mu0) ** 2  #(equation 14)

        if variance > max_variance: # find the maximum variance and corresponding threshold
            max_variance = variance
            threshold = i

    return threshold


#wrapper
def otsu_segment(image):
    threshold = otsu(image)
    segmented_image = (image >= threshold).astype(np.uint8)
    return segmented_image
















################## Part 2 #################################################











