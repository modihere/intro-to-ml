import pytesseract
from pytesseract import Output
import cv2
img = cv2.imread('/home/abzooba/PycharmProjects/intro-to-ml/deep_learning/OCR/first_page.png')
img = cv2.resize(img, (600, 700), interpolation=cv2.INTER_AREA)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
print(d)
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)