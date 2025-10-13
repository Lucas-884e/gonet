import gzip
import pickle
import numpy as np


def merge_X_y(X, y):
    return np.array([np.append(a, b) for a, b in zip(X, y)])


def export_csv(fname, dataset):
    m = merge_X_y(dataset[0], dataset[1])
    np.savetxt(fname, m, delimiter=',', fmt='%f')


if __name__ == '__main__':
    f = gzip.open("mnist.pkl.gz", 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()

    export_csv('training_data.csv', training_data)
    export_csv('validation_data.csv', validation_data)
    export_csv('test_data.csv', test_data)
