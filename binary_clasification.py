"""
Clasification problems expects outputs between 0 and 1
to compute that ouput use sigmoid function where expects a z logit

sigmoid = G(z) = 1/1+e-z
to get z use fw,b(x) = w*x + b
w and x being vectors



"""

import numpy as np
class BinaryClasification:
    def __init__(self):
        self.X = np.array([0,0,0,1,1,1,1])
        self.w = np.zeros(self.X.shape[0])
        self.b = 0
        self.threesold = 0.5

    def compute_z(self):return np.dot(self.X,self.w) + self.b
    
    def compute_sigmoid(self):
        ans =  1/(1 + np.exp(-self.compute_z()))      
        return 0 if ans<=self.threesold else 1
       



bn = BinaryClasification()


print(bn.compute_sigmoid())
