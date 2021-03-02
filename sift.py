import cv2 as cv
import numpy as np


def harris_detection(image):
    grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(grey_image, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(grey_image, cv.CV_64F, 0, 1, ksize=5)
    alpha = 0.06
    threshold = 619623619.2  # Max for Yosemite1.jpg = 2919623619.2

    sobel_x_square = np.multiply(sobel_x, sobel_x)
    sobel_y_square = np.multiply(sobel_y, sobel_y)
    sobel_xy_square = np.multiply(sobel_x, sobel_y)

    responses = np.zeros((image.shape[0], image.shape[1]))
    key_points = []

    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            neighbourhoodx = sobel_x[y - 2: y + 3, x - 2:x + 3]
            neighbourhoody = sobel_y[y - 2: y + 3, x - 2:x + 3]

            eigenIx = np.sum(neighbourhoodx)
            eigenIy = np.sum(neighbourhoody)

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
            else:
                responses[y, x] = 0

    for y in range(5, image.shape[0]):
        for x in range(5, image.shape[1]):
            if harris_max_suppress[y, x] != 0:
                key_points.append(cv.KeyPoint(x, y, 1))

    key_points_image = cv.drawKeypoints(image, key_points, image)
    print(key_points)

    return key_points, key_points_image


def perform_sift(image1, keypoints1, image2, keypoints2):
    keypoint_pixels1 = cv.KeyPoint_convert(keypoints1)
    keypoint_pixels2 = cv.KeyPoint_convert(keypoints2)



    for keypoint in keypoints1:
        xcoord = int(keypoint.pt[0])
        ycoord = int(keypoint.pt[1])
        neighborhood_16 = image1[ycoord - 7: ycoord + 9, xcoord - 7:xcoord + 9]
        print("Neighborhood size", neighborhood_16.shape)

        # Obtaining the 16 cells in the neighborhood
        cells = []
        for y in range(1, neighborhood_16.shape[0], 4):
            for x in range(1, neighborhood_16.shape[1], 4):
                cells.append(neighborhood_16[y - 1: y + 3, x - 1: x + 3])



original_image1 = cv.imread("Yosemite1.jpg")
original_image2 = cv.imread("Yosemite2.jpg")

def main():
    keypoints1, keypoint_image1 = harris_detection(original_image1)
    keypoints2, keypoint_image2 = harris_detection(original_image2)

    two_images = np.concatenate((keypoint_image1, keypoint_image2), axis=1)

    original_grey_image1 = cv.cvtColor(original_image1, cv.COLOR_BGR2GRAY)
    original_grey_image2 = cv.cvtColor(original_image2, cv.COLOR_BGR2GRAY)

    perform_sift(original_grey_image1, keypoints1, original_grey_image2, keypoints2)

    cv.imshow("Keypoints Image 1", keypoint_image1)
    cv.imshow("Keypoints Image 2", keypoint_image2)
    cv.imshow("Features", two_images)
    cv.waitKey(0)


if __name__ == '__main__':
    main()