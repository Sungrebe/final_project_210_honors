import numpy as np
import cv2
from scipy import signal, ndimage
from numpy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt

# Create a gaussian target, with the same shape as x
# Should be largest at i=0 and smallest at i=len(x)/2
def create_gaussian_target1D(x, sigma):
    # create a gaussian distribution centered at len(x)/2, use np.roll to shift it
    # so that the maximum is at i=0 and the second highest peak is at i=N-1
    return np.roll(np.exp(-(x - len(x)/2)**2/(sigma**2)), len(x)/2)

# Return w, linear correlation filter for x and y
# y will be defined by your created gaussian function
def linear_correlation1D(x, sigma, lambda_val):
    # based on the formula from the paper
    y = create_gaussian_target1D(x, sigma)
    return ( np.multiply(np.conjugate(fft(x)), fft(y)) ) / ( np.multiply(np.conjugate(fft(x)), fft(x)) + lambda_val )

# return the index where the correlation filter returns the 
# highest value for z
def detect1D(w, z):
    # get the index corresponding to the highest value for z
    # use multiplication in the frequency domain to perform convolution
    return np.argmax(ifft(w * fft(z)))

# Extend your previous function to 2D
def create_gaussian_target(x, sigma):
    gaussian_2d = np.zeros(np.shape(x))

    # compute the 2d gaussian using the formula e^-(x^2 + y^2)/(sigma**2)
    for i in range(len(x)):
        for j in range(len(x[0])):
            gaussian_2d[i][j] = np.exp(-((i - len(x)//2 - 1)**2 + (j - len(x)//2 - 1)**2)/(sigma**2))

    # shift so that the distribution is centered at (0, 0), with the next highest peaks at (0, N - 1), (N - 1, N - 1),
    # and (N - 1, 0)
    return np.roll(gaussian_2d, (x.shape[1]//2, x.shape[0]//2), axis=(0, 1))

# Extend your previous function to 2D
def linear_correlation(x, sigma, lambda_val):
    # essentially the same as the 1D formula, use fft2 instead of fft
    y = create_gaussian_target(x, sigma)
    return ( np.multiply(np.conjugate(fft2(x)), fft2(y)) ) / ( np.multiply(np.conjugate(fft2(x)), fft2(x)) + lambda_val )

# Extend your previous function to 2D
def detect(w, z):
    # same as 1D, since we are working with a 2D array use np.unravel_index to get the tuple corresponding
    # to the index of the maximum value for z
    flattened_conv = ifft2(w * fft2(z)).flatten()
    index = np.where(flattened_conv == np.max(flattened_conv))
    return np.unravel_index(index, np.shape(z))

# TODO Modify your original correlation filter to account for linear noise type 1
def unblur1(x):
    # use a Mean Filter to reduce noise

    blur = cv2.blur(x, (5, 5))

    plt.subplot(121),plt.imshow(x),plt.title("Original")
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title("Averaging")
    plt.xticks([]), plt.yticks([])
    cv2.imwrite("meanfiltered.jpg",blur)

    # uncomment this line to visualize the mean filter effect on the ROI
    #plt.show()

    return blur

# TODO Modify your original correlation filter to account for linear noise type 2
def unblur2(x):
     # I found a Gaussian Filter works reasonably well for this case

     blur = cv2.GaussianBlur(x, (5, 5), 1)

     plt.subplot(121),plt.imshow(x),plt.title("Original")
     plt.xticks([]), plt.yticks([])
     plt.subplot(122),plt.imshow(blur),plt.title("Averaging")
     plt.xticks([]), plt.yticks([])
     cv2.imwrite("gaussfiltered.jpg",blur)

     # uncomment this line to visualize the gaussian filter effect on the ROI
     #plt.show()

     return blur