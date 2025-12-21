"""
Clasification problems expects outputs between 0 and 1
to compute that ouput use sigmoid function where expects a z logit

sigmoid = G(z) = 1/1+e-z
to get z use fw,b(x) = w*x + b
w and x being vectors



"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
class BinaryClasification:
    def __init__(self):
        self.X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
        self.y = np.array([0,0,0,1,1,1]).reshape(-1,1)#reshape asserts 2 dimension array for predictions
        self.n,self.m = self.X.shape
        self.b = -3
        self.w = np.zeros(self.m)#zeros with len features
    def compute_z(self)->NDArray:return np.dot(self.X,self.w) + self.b
    def compute_sigmoid(self,z:NDArray)->NDArray:return 1/(1+np.exp(-z))
    
    def compute_cost_function(self,predictions):
        losses = -self.y*np.log(predictions) - (1-self.y)*np.log(1 - predictions)
        return np.mean(losses.sum())#same as 1/self.m
        
        
    def __call__(self):
       self.w = np.array([1,1]) 
       z = self.compute_z()
       y = self.y.reshape(-1, 1)

       predictions = self.compute_sigmoid(z)
       return self.compute_cost_function(predictions)
bn = BinaryClasification()


print(bn())
