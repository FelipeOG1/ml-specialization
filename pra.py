


import numpy as np
import matplotlib.pyplot as plt
from numpy._core.fromnumeric import shape
np.set_printoptions(precision = 2,suppress = True)
class MultipleLinearRegression:
    def __init__(self):
        self.X = np.array([[2104, 5, 1, 45], 
                           [1416, 3, 2, 40], 
                           [852, 2, 1, 35]],dtype = float)

        self.y_train = np.array([460, 232, 178],dtype = float)
        self.m , self.n = np.shape(self.X)
        self.w = np.zeros(self.n)
        self.b = 0
        self.y_hat = lambda x:np.dot(x,self.w) + self.b
    
    
    def compute_cost_function(self): return (1/(2 * self.m)) * ((self.y_hat(self.X) - self.y_train) **2).sum()
    
    
    def __call__(self):
        rows,cols = np.shape(self.X)
        print(f"rows:{rows}, cols:{cols}")





MultipleLinearRegression()()
