import cv2
import numpy as np


class ImageProcessing:

    def __init__(self, image):
        self.image = image

    def resize_image(self):
        self.image = cv2.resize(self.image, (600, 800), interpolation=cv2.INTER_AREA)

    def thresholding(self, g_image, inp):
        new_image = g_image
        # new_image = cv2.medianBlur(self.image,1)
        thresh1 = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        thresh2 = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        images = [new_image, thresh1, thresh2]
        for img in images:
            cv2.imshow("images", img)
            cv2.waitKey(0)
        if inp == 1:
            return thresh2

    def straighten_image(self):
        self.resize_image()
        cv2.imshow("images", self.image)
        cv2.waitKey(0)
        gray_image = self.image
        # gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_not(gray_image)
        thresh = self.thresholding(gray_image, 1)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        print(angle)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (height, width) = self.image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.imshow("Rotated", rotated)
        cv2.waitKey(0)


if __name__ == '__main__':
    Image = cv2.imread('test2.jpg', 0)
    processing = ImageProcessing(Image)
    processing.straighten_image()