import cv2 as cv
import numpy as np


original_image = cv.imread("Yosemite1.jpg")

def main():
    cv.imshow("Keypoints", original_image)


if __name__ == '__main__':
    main()