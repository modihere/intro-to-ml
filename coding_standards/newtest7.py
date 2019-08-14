def cube(y):
    return y * y * y;


CUBE = lambda x: x * x * x
print(CUBE(5))

print(cube(5))

li = [10, 20, 30, 40, 50, 61, 70, 80, 90, 100]
final_list = list(filter(lambda x: (x % 2 != 0), li))
print(final_list)

li = [10, 20, 30, 40, 50, 61, 70, 80, 90, 100]
final_list = list(map(lambda x: x * 2, li))
print(final_list)

from functools import reduce

li = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
total = reduce((lambda x, y: x + y), li)
print(total)

import pandas as pd

matrix = [(1, 2, 3),
          (4, 5, 6),
          (7, 8, 9),
          (10, 11, 12),
          (13, 14, 15),
          (16, 17, 18)
          ]
dfObj = pd.DataFrame(matrix, columns=list('abc'))
new = dfObj.apply(lambda num: num + 5)


print('\n', dfObj.head(), '\n\n', new.head())

print('\n\n', dfObj.tail(), '\n\n', new.tail())
