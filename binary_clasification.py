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
        self.X = np.array([1,2,3])
        self.m = len(self.X)
        self.w = np.zeros(self.m)
        self.b = 0
        
    def compute_logistic_func(self):return self.X * self.w + self.b
        
    def compute_sigmoid(self,z):return 1/(1 + np.exp(-z))#np.exp elevate euler to each element of entry z
    
    def __call__(self):
        z = self.compute_logistic_func()
        return self.compute_sigmoid(z)
        


     

bc = BinaryClasification()
        
print(bc())
        
