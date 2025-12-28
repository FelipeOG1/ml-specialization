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
        self.m,self.n= self.X.shape
        self.b = -3
        self.w = np.zeros((self.n,1))#zeros with len features
        self.alpha = 0.0001
        self.lam = 200000#lambda value for regularization
        self.REG_COST_FUNCTION = self.lam/(2*self.m) * np.sum(np.square(self.w))
        self.REG_DERIVATIVE = (self.lam / self.m)*self.w
        
    def compute_z(self)->NDArray:return np.dot(self.X,self.w) + self.b
    def compute_sigmoid(self,z:NDArray)->NDArray:return 1/(1+np.exp(-z))
    def compute_w_derivative(self,predictions):return np.mean(np.dot(self.X.T,(predictions - self.y)))
    def compute_b_derivative(self,predictions):return np.mean(predictions - self.y) 
    def compute_losses(self,predictions): return -self.y*np.log(predictions) - (1-self.y)*np.log(1 - predictions)
    def compute_cost_function(self,predictions):return np.mean(self.compute_losses(predictions))
    
        
    def gradient_descent(self,epsilon = 1e-6,max_iterations = 20000):
        prev_cost = float('inf')
        cost = 0
        for i in range(max_iterations):
            predictions = self.compute_sigmoid(self.compute_z())
            dw = self.compute_w_derivative(predictions)  # Shape: (n_features,)
            db = self.compute_b_derivative(predictions)  # Shape: scalar
            self.w -= self.alpha *dw
            self.b -=self.alpha * db
            cost = self.compute_cost_function(predictions)
            if abs(prev_cost - cost) < epsilon:
                
                break
            prev_cost = cost
        return cost 

       
    def regularize_gradient_descent(self,epsilon = 1e-6,max_iterations = 20000):
        prev_cost = float('inf')
        cost = 0
        for i in range(max_iterations):
            predictions = self.compute_sigmoid(self.compute_z())
            dw = self.compute_w_derivative(predictions) + self.REG_DERIVATIVE
            db = self.compute_b_derivative(predictions) #b doesnt regularize
            self.w -= self.alpha *dw
            self.b -=self.alpha * db
            cost = self.compute_cost_function(predictions) + self.REG_COST_FUNCTION
            print(cost)
            if abs(prev_cost - cost) < epsilon:
                break
            prev_cost = cost
        return cost 

 
    
    def __call__(self):
        return self.regularize_gradient_descent()
bn = BinaryClasification()
print(bn())
