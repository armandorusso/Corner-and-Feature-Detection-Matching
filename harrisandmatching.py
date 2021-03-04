import cv2 as cv
import numpy as np
import math


def harris_detection_q3(image):
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


def calculate_image_gradients(image):
    magnitudes = np.zeros(image.shape)
    orientations = np.zeros(image.shape)

    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            sum_squares = np.square((image[y, x + 1] - image[y, x - 1])) + np.square(
                (image[y + 1, x] - image[y - 1, x]))
            magnitude = np.sqrt(sum_squares)
            orientation = math.atan2((image[y + 1, x] - image[y - 1, x]), (image[y, x + 1] - image[y, x - 1]))
            orientation = np.rad2deg(orientation)
            magnitudes[y, x] = magnitude
            orientations[y, x] = orientation

    return magnitudes, orientations


def perform_sift(image1, keypoints1, image2, keypoints2):
    keypoint_pixels1 = cv.KeyPoint_convert(keypoints1)
    keypoint_pixels2 = cv.KeyPoint_convert(keypoints2)
    keypoint_histogram1 = []
    keypoint_histogram2 = []
    keypoint_descriptors1 = []
    keypoint_descriptors2 = []

    magnitudes1, orientations1 = calculate_image_gradients(image1)
    magnitudes2, orientations2 = calculate_image_gradients(image2)

    keypoint_descriptors1, keypoint_coords1 = create_descriptor(keypoint_descriptors1, keypoint_histogram1, keypoints1, orientations1)
    keypoint_descriptors2, keypoint_coords2 = create_descriptor(keypoint_descriptors2, keypoint_histogram2, keypoints2, orientations2)

    best_matches = []

    for x in range(len(keypoint_descriptors1)):
        ssds = []
        best_match_index = 0
        for y in range(len(keypoint_descriptors2)):
            ssd = ((np.square(keypoint_descriptors1[x] - keypoint_descriptors2[y])).sum())
            ssds.append(ssd)
            if min(ssds) == ssd:
                best_match_index = y
        ssdmin = min(ssds)
        ssds.remove(min(ssds))
        ssdminprime = min(ssds)
        ssds.remove(min(ssds))
        ratio_distance = ssdmin / ssdminprime

        if ratio_distance < 0.8:
            best_matches.append(cv.DMatch(x, best_match_index, 0))

    matched_image = cv.drawMatches(image1, keypoints1, image2, keypoints2, best_matches, image1, flags=2)
    cv.imshow("Matches", matched_image)



def create_descriptor(keypoint_descriptors1, keypoint_histogram1, keypoints1, orientations1):
    keypoint_coords = []
    for keypoint in keypoints1:
        xcoord = int(keypoint.pt[0])
        ycoord = int(keypoint.pt[1])
        keypoint_coord = (xcoord, ycoord)  # Maybe reverse these?
        keypoint_coords.append(keypoint_coord)
        descriptor = np.array(list())

        neighborhood_16 = orientations1[ycoord - 7: ycoord + 9, xcoord - 7:xcoord + 9]
        print("Neighborhood size", neighborhood_16.shape)

        # Obtaining the 16 cells in the neighborhood
        for y in range(1, neighborhood_16.shape[0], 4):
            for x in range(1, neighborhood_16.shape[1], 4):
                cell = neighborhood_16[y - 1: y + 3, x - 1: x + 3]
                bins = np.zeros(8)
                for y1 in range(cell.shape[0]):
                    for x1 in range(cell.shape[1]):
                        orientation = cell[y1, x1]
                        index = int(orientation // 45)
                        bins[index] += 1

                keypoint_histogram1.append(bins)
                descriptor = np.concatenate((bins, descriptor), axis=None)

        # If the descriptor is not of 128 length, it means it's a feature at the edge of the image (meaning we can't
        # get the 16x16 window)
        if descriptor.shape[0] < 128:
            descriptor = np.resize(descriptor, (128))

        normal = np.linalg.norm(descriptor)
        descriptor = np.divide(descriptor, normal)
        threshold_descriptor = np.array(descriptor)

        for i in range(descriptor.shape[0]):
            if descriptor[i] < 0.2:
                threshold_descriptor[i] = descriptor[i]
            else:
                threshold_descriptor[i] = 0

        normal_again = np.linalg.norm(threshold_descriptor)
        threshold_descriptor = np.divide(threshold_descriptor, normal_again)
        keypoint_descriptors1.append(threshold_descriptor)

    return keypoint_descriptors1, keypoint_coords


# First, we find Ixx, Iyy, and Ixy. To do this, we find the sobel images and multiply them respectively Next step is
# to compute the sum of eigenvalues in the 5x5 window IN the Ix and Iy images respectively. Once we have that,
# that is our H matrix From there, we compute R, which will return a value From there, we check the value of R and
# see if its large. If it is, it means its a corner. If R is negative, its an edge. If it's small, its a flat surface
# We then have to threshold R; set it really high Then, to max suppress, all you gotta do is check the neighbourhood
# and see if that pixel is bigger than the neighbourhood. If it is, then keep it

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

    key_points_image = cv.drawKeypoints(original_image, key_points, original_image)

    cv.imshow("Keypoints Image", key_points_image)
    cv.imshow("Sobel X", sobel_x_square)
    cv.imshow("Sobel Y", sobel_y_square)
    cv.imshow("Sobel XY", sobel_xy_square)
    cv.imshow("Response", response_image)


original_image = cv.imread("Yosemite1.jpg")
original_image1 = cv.imread("Yosemite1.jpg")
original_image2 = cv.imread("Yosemite2.jpg")


def main():
    # Question 2
    # harris_detection(original_image)
    # cv.imshow("Keypoints", original_image)

    # Question 3
    keypoints1, keypoint_image1 = harris_detection_q3(original_image1)
    keypoints2, keypoint_image2 = harris_detection_q3(original_image2)

    two_images = np.concatenate((keypoint_image1, keypoint_image2), axis=1)

    original_grey_image1 = cv.cvtColor(original_image1, cv.COLOR_BGR2GRAY)
    original_grey_image2 = cv.cvtColor(original_image2, cv.COLOR_BGR2GRAY)

    best_matches = perform_sift(original_grey_image1, keypoints1, original_grey_image2, keypoints2)

    cv.imshow("Keypoints Image 1", keypoint_image1)
    cv.imshow("Keypoints Image 2", keypoint_image2)
    cv.imshow("Features", two_images)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
