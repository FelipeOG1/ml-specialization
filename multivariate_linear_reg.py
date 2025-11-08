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
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]],dtype = float)
        self.y_train = np.array([460, 232, 178],dtype = float)
        self.y_mean = np.mean(self.y_train)
        self.y_std = np.std(self.y_train)
        self.y_train = (self.y_train - self.y_mean) / self.y_std
        self.m,self.n = self.x_train.shape
        self.b = 0
        self.w = np.zeros(self.n)
        self.y_hat = lambda x_train:np.dot(x_train,self.w) + self.b 
        self.learning_rate = 5.0e-7
        

    def compute_w_derivate(self):
        hat_minus_train = self.y_hat(self.x_train) - self.y_train
        return (1 / self.m) * np.dot(self.x_train.T , (hat_minus_train)) , hat_minus_train #avoid computing y_hat - y_train twice
    def compute_b_derivate(self):return ((self.y_hat(self.x_train) - self.y_train)).sum() / self.m #this is equivalent to doing np.mean(self.y_hat - self.y_train)
    def compute_cost_function(self):return (1 / (2 * self.m)) * np.sum((self.y_hat(self.x_train) - self.y_train) ** 2)

    def gradient_descent(self,epsilon = 0.000001,max_iterations = 1000000):
        prev_cost = self.compute_cost_function()
        for i in range(max_iterations):
            w_derivative,hat_minus_train = self.compute_w_derivate()
            self.w = self.w - self.learning_rate * w_derivative
            self.b = self.b - self.learning_rate * np.mean(hat_minus_train)

            current_cost = self.compute_cost_function()
            print(current_cost,prev_cost)
            if abs(current_cost - prev_cost) < epsilon:
                print(f"convergence on {i}th iteration")
                break    
            prev_cost = current_cost
            
          
        print(self.compute_cost_function())
MultipleLinearRegression().gradient_descent()
