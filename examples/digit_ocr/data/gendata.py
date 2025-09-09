import csv
from sklearn.datasets import load_digits


digits = load_digits()

dataset = []
for x, y in zip(digits.data, digits.target):
    d = x.tolist()
    d.append(y.item())
    dataset.append(d)


with open('digits.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataset)
