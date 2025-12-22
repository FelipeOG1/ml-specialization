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
        self.alpha = 0.0001
        
    def compute_z(self)->NDArray:return np.dot(self.X,self.w) + self.b
    def compute_logistic_function(self,z:NDArray)->NDArray:return 1/(1+np.exp(-z))
    def compute_w_derivative(self,predictions):return np.mean(np.dot(self.X.T,(predictions - self.y)))
    def compute_b_derivative(self,predictions):return np.mean(predictions - self.y) 
    def compute_losses(self,predictions): return -self.y*np.log(predictions) - (1-self.y)*np.log(1 - predictions)
    def compute_cost_function(self,predictions):return np.mean(self.compute_losses(predictions))
        

    def gradient_descent(self,epsilon = 1e-6,max_iterations = 10000):
        predictions = self.compute_logistic_function(self.compute_z())
        prev_cost = self.compute_cost_function(predictions)
        new_cost = 0
        for _ in range(max_iterations):
            
            self.w = self.alpha - self.compute_w_derivative(predictions)
            self.b = self.alpha - self.compute_b_derivative(predictions)
            new_cost = self.compute_cost_function(self.compute_logistic_function(self.compute_z()))
            if abs(new_cost - prev_cost)<=epsilon:
                prev_cost = new_cost
                break
            
            
            
    @property
    def features(self):
       pass
        
    def __call__(self):
        predictions = self.compute_logistic_function(self.compute_z())
        assert(predictions.shape[0]  == self.X.shape[0])
        return self.compute_w_derivative(predictions)
bn = BinaryClasification()
print(bn())
