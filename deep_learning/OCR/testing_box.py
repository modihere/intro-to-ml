import cv2
import numpy as np

large = cv2.imread('test4.png')
rgb = cv2.pyrDown(large)
rgb = large
# rgb = cv2.resize(rgb, (600, 800), interpolation=cv2.INTER_AREA)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', small)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('thresh', grad)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
cv2.imshow('thresh', connected)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)

print(len(contours))
count = 0
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y + h, x:x + w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)

    r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

    if r > 0.35 and r < 95.00 and w > 10 and h > 7:
        count += 1
        cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
print(count)
cv2.imshow('rects', rgb)
cv2.waitKey(0)
