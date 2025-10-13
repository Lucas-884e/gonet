import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


lineno = int(sys.argv[1])


data = None
with open('digits.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for i in range(lineno-1):
        next(reader)
    print(f'skip {lineno-1} rows')
    for row in reader:
        data = row
        break


print(f'Plot digit {int(data[-1])}')
image = np.array(data[:-1]).reshape((8, 8))
plt.matshow(image, cmap="gray")
plt.show()
