# Implementation of least square
# Luca Iocchi, 2014-2018

import numpy as np

class LeastSquare:

    def __init__(self):
        self.w = [0, 0, 0]

    def fit(self,X,t):
        n = len(X) # nr. of examples
        t2 = np.c_[t, 1-t]
        phi = np.c_[np.ones(n), X] # desing matrix
        self.w = np.matmul(np.linalg.pinv(phi),t2) # Least square solution
        print "Least square solution:"
        print self.w.transpose()

    def predict(self,x):
        xn = np.array((1, x[0][0], x[0][1]))
        yn = np.matmul(self.w.transpose(),xn)
        if yn[0]>yn[1]:
            return 1
        else:
            return -1

