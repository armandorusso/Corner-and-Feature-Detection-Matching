import cv2 as cv
import numpy as np


def create_gaussian_kernel(sigma):
    kernel_size = int(sigma * 3)

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size), np.float32)  # Might cause a bug with the float?
    rows = kernel.shape[0]
    cols = kernel.shape[1]

    for x in range(rows):
        for y in range(cols):
            kernel[x, y] = ((1 / (2 * np.pi * (sigma ** 2)))) * (np.e ** (-(x ** 2 + y ** 2 / (2 * (sigma ** 2)))))
            print('Kernel:', kernel[x, y])

    sum = np.sum(np.sum(kernel, 1), 0)
    return kernel, sum;


def apply_gaussian_blur(image, factor):
    kernel, summation = create_gaussian_kernel(factor)

    outputarray = np.array(image)
    summation = float(summation)

    for i in range(1):
        outputarray = cv.filter2D(outputarray, -1, kernel)

    outputarray = cv.normalize(outputarray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    return outputarray


def create_sobel_image(image, kernel, greyscale, sobel_question):
    if greyscale:
        outputarray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    else:
        outputarray = image

    kernel = np.multiply(kernel, 1 / 8)  # Used to remove noise from the image

    outputarray = np.array(outputarray, dtype=np.float64)
    outputarray = cv.filter2D(outputarray, -1, kernel)

    if sobel_question:
        outputarray = cv.normalize(outputarray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
        outputarray = np.array(outputarray, dtype=np.uint8)

    return outputarray


def apply_sobel_x(image, greyscale, sobel_question):
    kernel = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]], dtype=np.float64
    )

    outputarray = create_sobel_image(image, kernel, greyscale, sobel_question)

    return outputarray


def apply_sobel_y(image, greyscale, sobel_question):
    kernel = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]], dtype=np.float64
    )

    outputarray = create_sobel_image(image, kernel, greyscale, sobel_question)

    return outputarray

# First, we find Ixx, Iyy, and Ixy. To do this, we find the sobel images and multiply them respectively Next step is
# to compute the sum of eigenvalues in the 5x5 window IN the Ix and Iy images respectively. Once we have that,
# that is our H matrix From there, we compute R, which will return a value From there, we check the value of R and
# see if its large. If it is, it means its a corner. If R is negative, its an edge. If it's small, its a flat surface
# We then have to threshold R; set it really high Then, to max suppress, all you gotta do is check the neighbourhood
# and see if that pixel is bigger than the neighbourhood. If it is, then keep it

def harris_detection(image):
    sobel_x = apply_sobel_x(image, greyscale=True, sobel_question=True)
    sobel_y = apply_sobel_y(image, greyscale=True, sobel_question=True)
    alpha = 0.05
    threshold = 10287891.2  # Max for Yosemite1.jpg

    sobel_x_square = np.multiply(sobel_x, sobel_x)
    sobel_y_square = np.multiply(sobel_y, sobel_y)
    sobel_xy_square = np.multiply(sobel_x, sobel_y)

    gauss_sobel_x_square = apply_gaussian_blur(sobel_x_square, 3)
    gauss_sobel_y_square = apply_gaussian_blur(sobel_y_square, 3)
    gauss_sobel_xy_square = apply_gaussian_blur(sobel_xy_square, 3)
    responses = np.zeros((image.shape[0], image.shape[1]))
    key_points = []

    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            neighbourhoodx = gauss_sobel_x_square[y - 2: y + 3, x - 2:x + 3]
            neighbourhoody = gauss_sobel_y_square[y - 2: y + 3, x - 2:x + 3]

            eigenIx = np.sum(neighbourhoodx)
            eigenIy = np.sum(neighbourhoody)

            harris_matrix = np.array([[eigenIx, 0], [0, eigenIy]])
            response = (eigenIx * eigenIy) - alpha * ((eigenIx + eigenIy) ** 2)

            if response > threshold:
                responses[y, x] = response

    response_image = responses / responses.max() * 255
    response_image = response_image.astype(np.uint8)

    # Max Suppression
    harris_max_suppress = np.zeros((image.shape[0], image.shape[1]))
    print("Harris Shape", harris_max_suppress.shape)

    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            neighbourhood = responses[y - 2: y + 3, x - 2:x + 3]
            if responses[y, x] == neighbourhood.max():
                harris_max_suppress[y, x] = responses[y, x]

    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            if harris_max_suppress[y, x] != 0:
                key_points.append(cv.KeyPoint(x, y, 1))

    key_points_image = cv.drawKeypoints(original_image, key_points, original_image)

    cv.imshow("Keypoints Image", key_points_image)
    cv.imshow("Sobel X", gauss_sobel_x_square)
    cv.imshow("Sobel y", gauss_sobel_y_square)
    cv.imshow("Sobel xy", gauss_sobel_xy_square)
    cv.imshow("Response", response_image)
    cv.imshow("Harris Non-Max", harris_max_suppress)
    cv.waitKey(0)


original_image = cv.imread("Yosemite1.jpg")


def main():
    harris_detection(original_image)
    cv.imshow("Keypoints", original_image)


if __name__ == '__main__':
    main()
