import numpy as np

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_num in range(classes):
        dx = range(points*class_num, points*(class_num+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_num*4, (class_num+1)*4, points) + np.random.randn(points)*.2
        X[dx] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[dx] = class_num
    return X, y