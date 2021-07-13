#  Program used to manipulate images by performing
#  convolution between them, noise and filters.
#
#  Dekas Dimitrios
#  Digital Image Processing
#  Assignment A 2020
#  version: 6.1
#  date: 1/11/2020 6:36PM

import numpy as np
import cv2
# import scipy.signal as sps
# from skimage.util import random_noise


# Function used to display an image and wait for key input to continue.
def showImageAndWait(title, image):
    # Show Image
    cv2.imshow(title, image)

    # Wait Key Press
    cv2.waitKey(0)

    # Close Image Window
    cv2.destroyAllWindows()


# Function used to rotate a given array, 180 degrees.
def flip(arr):
    flippedArr = np.array(arr)  # Creates an array with the same shape as arr
    (rows, cols) = flippedArr.shape[:2]
    for i in range(rows):
        for j in range(cols):
            flippedArr[i][cols - 1 - j] = arr[rows - 1 - i][j]  # Makes 180 Degrees Rotation
    return flippedArr


# Implementation of the 2D convolution between A and B.
# Argument 'param' used to determine the output's array shape.
def myConv2D(A, B, param):
    if A.shape[0] > B.shape[0]:  # Use Smallest Array as Kernel
        h = flip(B)
        x = A
    else:
        h = flip(A)
        x = B
    (xRows, xCols) = x.shape[:2]
    (hRows, hCols) = h.shape[:2]

    rowCenterIdx = hRows // 2  # Kernel h Central Row Index
    colCenterIdx = hCols // 2  # Kernel h Central Col Index

    if param == 'same':  # Param Determines Output Shape
        y = np.zeros((xRows, xCols), int)
        rowGap = rowCenterIdx  # Determines max unused rows when performing convolution
        colGap = colCenterIdx  # Determines max unused cols when performing convolution
    else:
        y = np.zeros((xRows + hRows - 1, xCols + hCols - 1), int)
        rowGap = 2 * rowCenterIdx  # Determines max unused rows when performing convolution
        colGap = 2 * colCenterIdx  # Determines max unused rows when performing convolution

    (yRows, yCols) = y.shape[:2]
    for i in range(yRows):  # Output y Rows
        for j in range(yCols):  # Output y Cols
            for m in range(hRows):  # Kernel h Rows
                for n in range(hCols):  # Kernel h Cols
                    xi = i + (m - rowGap)  # Transform y Row Index To x Row Index
                    xj = j + (n - colGap)  # Transform y Col Index To x Col Index
                    if 0 <= xi < xRows and 0 <= xj < xCols:  # Check x Indexes Validity
                        y[i][j] += x[xi][xj] * h[m][n]  # Use Convolution's Formula
    return y


# Function used to add gaussian noise to the given image.
def myGaussianNoise(image):
    # Create an array with random values that follow the gaussian distribution
    # with mean = 0 and standard deviation = 0.1, one for each image pixel.
    gaussian_noise = np.random.normal(0, 0.1, image.shape)
    # Normalize the image so it fits the range 0-1,
    # after converting it to a float array.
    image = image.astype('float64')
    image /= image.max()
    # Add the produced noise to the image.
    noisy_image = np.add(image, gaussian_noise)
    # Scale up the image to the range 0-255
    noisy_image *= 255
    # Create a np array of data type 'uint8' filled with the scaled up values.
    output = np.array(noisy_image, dtype='uint8')
    return output


# Function used to add salt and pepper noise to the given image.
def mySaltAndPepperNoise(image):
    # Create an array with random values, one for each image pixel.
    prob = np.random.random(image.shape[:2])
    # Determine the probability threshold for introducing white and black pixels in the image.
    threshold = 0.05
    # Each pixel with value lower than threshold at prob array will become a black pixel.
    image[prob < threshold] = 0
    # Each pixel with value higher than 1 - threshold at prob array will become a white pixel.
    image[prob > 1 - threshold] = 255
    return image


# Function used to process an image using a mean filter.
def myMeanFilter(image, fSize):
    # Create mean array
    mean = np.ones((fSize, fSize))
    mean /= np.sum(mean)
    # Convolve mean array with image in order to complete filtering
    mean_image = myConv2D(image, mean, 'same')
    # Return the result of the convolution in the type format needed for cv2.imshow().
    return mean_image.astype(np.uint8)


# Function used to process an image using a median filter.
def myMedianFilter(image, fSize):

    # Create the output and window arrays
    median_image = np.zeros(image.shape)
    window = np.zeros(fSize * fSize)

    # Window Central Row Index
    rowGap = fSize // 2
    # Window Central Column Index
    colGap = fSize // 2

    # Use zero padding in the outside layer.
    for i in range(rowGap, image.shape[0] - rowGap):
        for j in range(colGap, image.shape[1] - colGap):
            # For each element of the window
            for x in range(fSize):
                for y in range(fSize):
                    # Calculate its value
                    window[(x * fSize) + y] = image[i + x - rowGap][j + y - colGap]
            # Get the window values sorted
            window = sorted(window)
            # The (i,j)th value of the median image is the middle element of the sorted window
            median_image[i][j] = window[(fSize * fSize) // 2]

    # Return the result in the type format needed for cv2.imshow().
    return median_image.astype(np.uint8)


# Function used to add noise to a given image. Noise chosen based on the value of argument 'param'.
# 'param' options: 'gaussian' and 'saltandpepper'.
def myImNoise(image, param):
    if param == 'gaussian':
        return myGaussianNoise(image)
    elif param == 'saltandpepper':
        return mySaltAndPepperNoise(image)
    else:
        return print('Error! Wrong parameter given. Choose between \'gaussian\' and \'saltandpepper\'.')


# Function used to filter a given image. Filter chosen based on the value of argument 'param'.
# 'param' options: 'mean' and 'median'.
def myImFilter(image, param, fSize):
    if param == 'mean':
        return myMeanFilter(image, fSize)
    elif param == 'median':
        return myMedianFilter(image, fSize)
    else:
        return print('Error! Wrong parameter given. Choose between \'mean\' and \'median\'.')


if __name__ == '__main__':
    # Checks myConv2D validity.
    # A = np.array([[25, 100, 75, 49, 130, 22, 80, 1], [50, 80, 0, 70, 100, 86, 5, 30], [5, 10, 20, 30, 0, 5, 35, 11],
    #               [60, 50, 12, 24, 32, 7, 60, 9], [37, 53, 55, 21, 90, 162, 3, 59], [140, 17, 0, 23, 222, 95, 12, 1],
    #               [10, 7, 43, 67, 22, 955, 122, 324], [40, 137, 205, 233, 62, 25, 13, 54]])
    # B = np.array([[1, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0]])
    # print(myConv2D(A, B, 'full') == sps.convolve2d(A, B, 'full'))
    # print(myConv2D(A, B, 'same') == sps.convolve2d(A, B, 'same'))
    # A = np.array([[25, 100, 75, 49, 130, 12], [50, 80, 0, 70, 100, 1], [5, 10, 20, 30, 0, 56], [60, 50, 12, 24, 32, 87],
    #               [37, 53, 55, 21, 90, 3], [140, 17, 0, 23, 222, 473]])
    # B = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    # print(myConv2D(A, B, 'full') == sps.convolve2d(A, B, 'full'))
    # print(myConv2D(A, B, 'same') == sps.convolve2d(A, B, 'same'))
    # Seems to work properly. I hope so at least.

    # Read image from disk and convert it to black and white.
    A = cv2.imread('test.jpg', 0)
    showImageAndWait('Image', A)

    # Add gaussian noise to the image.
    gaussian = myImNoise(A, 'gaussian')
    showImageAndWait('Image with Gaussian', gaussian)

    # Use mean filter to temper the effect of gaussian noise.
    meanGaussian = myImFilter(gaussian, 'mean', 3)
    showImageAndWait('Mean Filtered Image with Gaussian', meanGaussian)

    # Use median filter to temper the effect of gaussian noise.
    medianGaussian = myImFilter(gaussian, 'median', 3)
    showImageAndWait('Median Filtered Image with Gaussian', medianGaussian)

    # Add Salt and Pepper noise to the image.
    sap = myImNoise(A, 'saltandpepper')
    showImageAndWait('Image with Salt and Pepper', sap)

    # Use mean filter to temper the effect of salt and pepper noise.
    meanSap = myImFilter(sap, 'mean', 3)
    showImageAndWait('Mean Filtered Image with Salt and Pepper', meanSap)

    # Use median filter to temper the effect of salt and pepper noise.
    medianSap = myImFilter(sap, 'median', 3)
    showImageAndWait('Median Filtered Image with Salt and Pepper', medianSap)
