#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt


data_file = 'data.csv'
data_x, data_y = [], []

with open(data_file, newline='') as fdata:
  reader = csv.reader(fdata, delimiter=',', quotechar='"')
  for i, [xa, ya, xb, yb] in enumerate(reader):
    if i > 20000:
      break
    data_x.extend([float(xa), float(xb)])
    data_y.extend([float(ya), float(yb)])


plt.plot(data_x, data_y, 'ro')
plt.axis([-15, 25, -10, 15])
plt.show()
