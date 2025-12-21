"""
Notation

Uni means only one feature

x_train = house size 
y_train = house price

x(i),y(i) = ith example on dataset

m = Number of training examples

w:parameter = weight
b:parameter = bias

f_wb = result of evaluation x(i) also known as "y hat" prediction

f(x) = wx + b 

"""
from typing import dataclass_transform
import numpy as np
import matplotlib.pyplot as plt

class UniLinearRegression:
    def __init__(self,learning_rate:float):
        np.random.seed(0)
        self.m = 50
        self.x_train = np.linspace(0, 10, self.m)                  
        self.y_train = 3 * self.x_train + 4 + np.random.randn(self.m) * 2 
        self.w = 0
        self.b = 0 
        self.y_hat = lambda x_train: self.w * x_train + self.b
        self.learning_rate = learning_rate 
        self.mean_normalization = lambda train: (train - np.mean(train)) / (np.max(train) - np.min(train))

        
    def compute_cost_function(self):return ((self.y_hat(self.x_train) - self.y_train) ** 2).sum() / (2 * self.m)
    def compute_w_derivative(self):return ((self.y_hat(self.x_train) - self.y_train) * self.x_train).sum() / self.m
    def compute_b_derivative(self):return (self.y_hat(self.x_train) - self.y_train).sum() / self.m


    def gradient_descent(self,max_iter = 100000,epsilon = 1e-6,prev_cost = float("inf")):
        for i in range(max_iter):
            temp_w = self.w - self.learning_rate * self.compute_w_derivative()
            temp_b = self.b - self.learning_rate * self.compute_b_derivative()
            self.w,self.b = temp_w,temp_b
            current_cost = self.compute_cost_function()
            print(f"prev_cost:{prev_cost},current_cost:{current_cost}")          
            if abs(current_cost - prev_cost) < epsilon:
                print(f"w and b converge on {i} iteration")
                break
            prev_cost = current_cost

    def compute_model_output(self):
        f_wb = np.zeros(self.m)
        for i in range(self.m):
            f_wb[i] = self.y_hat(self.x_train[i])
        return f_wb

    def compute_espec_size(self,size:float):
        self.gradient_descent()
        return self.w * size + self.b
    
    def _draw_graph(self,tmp_f_wb):
        plt.plot(self.x_train,tmp_f_wb, c = 'b',label = 'our Prediction')
        plt.scatter(self.x_train, self.y_train, marker='x', c='r',label='Actual Values')
        plt.title("Housing Prices")
        plt.ylabel('Price (in 1000s of dollars)')
        plt.xlabel('Size (1000 sqft)')
        plt.savefig("graph.png")
    
    def main(self):
        self.gradient_descent()
        self._draw_graph(self.compute_model_output())
        
    def __call___(self):
        self.gradient_descent()
        self._draw_graph(self.compute_model_output())  
        
ulr = UniLinearRegression(0.0001)
ulr.main()
