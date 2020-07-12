# Implementation of Fisher's discriminant 
# Luca Iocchi, 2014-2018

import numpy as np

class FisherDiscriminant:

    def __init__(self):
        self.w = [0, 0, 0]
        self.label = "Fisher Discriminant"

    def fit(self,X,t):
        n = len(X)  # num of examples
        # group the two subsets 
        # C1 = positive samples, C2 = negative samples
        C1 = np.ndarray((0,2))
        C2 = np.ndarray((0,2))
        for i in range(0,len(X)):
            if (t[i][0] == 1):
                C1 = np.vstack([C1, [X[i,0],X[i,1]]])
            else:
                C2 = np.vstack([C2, [X[i,0],X[i,1]]])			
        
        # compute means m1, m2
        m1 = np.mean(C1, axis=0)
        m2 = np.mean(C2, axis=0)
        
        # compute covariances S1, S2
        S1 = np.zeros((2,2))
        d = np.array(())
        for c in C1:
            d = np.subtract(c,m1).reshape(2,1)
            dt = d.transpose()
            S1 = S1 + np.matmul(d,dt)
        
        S1 = S1/len(C1);
        
        S2 = np.zeros((2,2))
        for c in C2:
            d = np.subtract(c,m2).reshape(2,1)
            dt = d.transpose()
            S2 = S2 + np.matmul(d,dt)
        S2 = S2/len(C2);
        
        # compute Sw matrix
        Sw = S1+S2
        
        # compute solution w 
        wt = np.matmul(np.linalg.inv(Sw),(m1-m2))
        
        # global mean
        mu = m1 * 0.5 + m2 * 0.5
        
        # compute constant term
        w0 = np.dot(wt,mu)
        
        # format the final solution
        self.w = np.array([-w0, wt[0], wt[1]])
        print "Fisher discriminant solution:"
        print self.w.transpose()

    
    def predict(self,x):
        xn = np.array((1, x[0][0], x[0][1]))
        yn = np.matmul(self.w.transpose(),xn)
        if yn>0:
            return 1
        else:
            return -1
                
                