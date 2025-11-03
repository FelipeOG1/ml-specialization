"""
Notation


x_train = house size 
y_train = house price

x(i),y(i) = ith example on dataset

m = Number of training examples

w:parameter = weight
b:parameter = bias

f_wb = result of evaluation x(i) also known as "y hat" prediction

f(x) = wx + b 

"""
import numpy as np
import matplotlib.pyplot as plt

class uniLinearRegression:
    def __init__(self):
        self.x_train = np.array([1.0,2.0])
        self.y_train = np.array([300.0,500.0])
        self.m = len(self.x_train)
        self.w = 200
        self.b = 100
        self.y_hat = lambda x: self.w * x + self.b


    def compute_model_output(self):
        f_wb = np.zeros(self.m)
        for i in range(self.m):
            f_wb[i] = self.w * self.x_train[i] + self.b
        return f_wb

    def compute_espec_size(self,size:float):
        return self.w * size + self.b
   
    def compute_cost_function(self):
        return sum([(self.y_hat(x) - y)**2 for x,y in zip(self.x_train,self.y_train)]) / (2 * self.m)

    

    def draw_graph(self):
        tmp_f_wb  = self.compute_model_output()
        
        plt.plot(self.x_train,tmp_f_wb, c = 'b',label = 'our Prediction')
        plt.scatter(self.x_train, self.y_train, marker='x', c='r',label='Actual Values')

        plt.title("Housing Prices")
        plt.ylabel('Price (in 1000s of dollars)')
        plt.xlabel('Size (1000 sqft)')
        plt.savefig("graph.png")




ulr = uniLinearRegression()
ulr.draw_graph()
print(ulr.compute_cost_function())
