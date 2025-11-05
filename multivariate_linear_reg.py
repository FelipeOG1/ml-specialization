"""
Multifeature Linear Regression


n = num_features
Xj = jth feature
   (i)
Xj     = value of feature j on ith example

->(i)
X     = ith training example (vector of training example)

->(2)
X     = second vector of training example  


Model:  fw,b(x) = w1x1 + ..... wnxn + b
    
Cost Function
  ->
J(w,b)



"""

import numpy as np
class MultipleLinearRegression:
    def __init__(self):
        self.x = np.array([
            [10, 20, 30],   
            [5,  15, 25],               
            [2,  4,  6],   
            [8,  16, 24]    
        ])
        self.w = np.array([1.0,2.5,-3.3])
        self.b = 4
        self.m = self.x.shape[0]
        self.y = np.array([460.0,232.0,315.0,178.0])
    
        

    def gradient_descent(self):
        #TODO COMPUTE DERIVATIE GOING FROM self.x.T[0] ... self.m
        pass

    def sum_vectorization(self):return np.dot(self.x,self.w) + self.b
    
           



ml = MultipleLinearRegression()

ml.sum_vectorization()
ml.gradient_descent()
