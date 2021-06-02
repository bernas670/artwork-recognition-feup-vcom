import cv2 as cv


def openImage(path):
    image = cv.imread(path)

    if image is None:
        print("  !! Error reading image ", path)
        return None

    return image
