import cv2

class ImageProcessing:

    def __init__(self):
        self.image = cv2.imread('first_page.png', 0)

    def thresholding(self):
        #cv2.imshow("input", self.image)
        new_image = self.image
        #new_image = cv2.medianBlur(self.image,1)
        thresh1 = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        images = [new_image, thresh1]
        for img in images:
            cv2.imshow("images", img)
            cv2.waitKey(0)
processing = ImageProcessing()
processing.thresholding()