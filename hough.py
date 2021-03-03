import cv2 as cv
import numpy as np
import math


def create_sobel_image(image, kernel, greyscale, sobel_question):
    if greyscale:
        outputarray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow('Original Grey image', outputarray)

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


def apply_canny(image):
    canny_image = np.array(image.shape)

    canny_image = cv.Canny(image, 50, 150)

    cv.imshow('Canny Edge Map', canny_image)

    return canny_image


def apply_hough(image):
    aside = np.square(image.shape[0])
    bside = np.square(image.shape[1])
    sum = aside + bside
    hypo = int(np.sqrt(sum))
    threshold = 15  # 15 for hough1.png, 100 for hough2.png

    hough_space = np.zeros((hypo, 180))
    polar_coordinate = []

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] != 0:
                for theta in range(0,
                                   180):  # Draw an imaginary line at each angle and update the vote count for that theta
                    theta_rad = np.deg2rad(theta)
                    d = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))  # Make sure its in radians
                    hough_space[d, theta] += 1
                    polar_coordinate.append((d, theta))

    print("Max Value in Hough Space: ", hough_space.max())

    for d in range(hough_space.shape[0]):
        for angle in range(hough_space.shape[1]):
            if hough_space[d, angle] > threshold:
                theta_in_radians = np.deg2rad(angle)
                aline = np.cos(theta_in_radians)
                bline = np.sin(theta_in_radians)
                x0 = aline * d
                y0 = bline * d
                x1 = int(x0 + 1000 * (-bline))
                y1 = int(y0 + 1000 * (aline))
                x2 = int(x0 - 1000 * (-bline))
                y2 = int(y0 - 1000 * (aline))
                cv.line(hough_image, (x1, y1), (x2, y2), (255, 100, 255), 1)

    print(hough_space)
    hough_graph = np.divide(hough_space, 255)
    hough_graph = np.array(hough_graph, dtype=np.float32)

    cv.imshow('Hough Space', hough_graph)


hough_image = cv.imread("hough1.png")


def main():
    # Question 1
    canny = apply_canny(hough_image)
    apply_hough(canny)
    cv.imshow('Hough Lines', hough_image)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
