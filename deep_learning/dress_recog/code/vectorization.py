import numpy as np
import time
a = np.random.rand(10000000)
b = np.random.rand(10000000)
tic = time.time()

c = np.dot(a, b)
d = np.dot([5, 5], [4, 4])
toc =time.time()
print("Vectorization version", str(1000*(toc-tic)), "ms")
print(c, d)
c = 0
tic = time.time()
for i in range(10000000):
    c += a[i]*b[i]
toc = time.time()

print("Non vectorized approach", str(1000*(toc-tic)), "ms")
print(c)

arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr)
cal = arr.sum(axis=0)
print(cal)
percentage = 100*arr//(cal.reshape(1,4))
print(percentage)