import cv2
import numpy as np

def sorted_contours(cnts):

    boundingboxes = [cv2.boundingRect(cnt) for cnt in cnts]
    (cnts,boundingboxes) = zip(*sorted(zip(cnts, boundingboxes), key=lambda b: b[1][1], reverse=False))
    return cnts, boundingboxes


large = cv2.imread('images/test7.jpg')
rgb = cv2.pyrDown(large)
rgb = large
rgb = cv2.resize(rgb, (600, 800), interpolation=cv2.INTER_AREA)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', small)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('grad', grad)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
cv2.imshow('thresh', connected)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)

print(len(contours))
count = 0
area_list = []

# sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
sorted_ctrs, boundingboxes = sorted_contours(contours)
print(len(sorted_ctrs))

for idx in range(len(sorted_ctrs)):
    x, y, w, h = cv2.boundingRect(sorted_ctrs[idx])
    mask[y:y + h, x:x + w] = 0
    cv2.drawContours(mask, sorted_ctrs, idx, (255, 255, 255), -1)

    r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

    if r > 0.35 and w > 5 and h > 5:
        count += 1
        cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        area_list.append(w * h)
        roi = rgb[y:y+h, x:x+w]
        cv2.imwrite('debug/file {}.png'.format(count), roi)
print(count)
print(area_list)
cv2.imshow('rects', rgb)
cv2.imwrite('debug/final {}.png'.format(count), rgb)
cv2.waitKey(0)
