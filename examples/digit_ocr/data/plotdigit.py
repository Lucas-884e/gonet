import csv
import matplotlib.pyplot as plt
import numpy as np


N = 5


lines = []
with open('digits.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        lines.append(row)


data = lines[N]
image = np.array(data[:-1]).reshape((8, 8))
plt.matshow(image, cmap="gray")
plt.show()
