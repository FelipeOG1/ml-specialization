import numpy as np
import tensorflow as tf


class Neuron:
    def compute_scalar(self,x,w,b):
        sigmoid = lambda z:1/(1+np.exp(-z))
        return sigmoid(-(np.dot(x,w)) + b)
    
class Layer:
    def __init__(self,units):
        self.units = units
        self.core = [Neuron() for _ in range(self.units)]
        self.a_out = np.zeros(units)
        self.w =  np.array([[1,-3,5],[2,4,-6]],dtype = np.float32)#fix w example
        self.b = np.array([-1,1,2],dtype = np.float32)
        
    def __call__(self,x):
        predictions = np.array([
            self.core[i].compute_scalar(self.w[:,i],self.b[i],x) for i in range(len(self.core))])
        self.a_out = predictions
        return self.a_out
        
    def set_weights(self,w,b):self.w,self.b = w,b
            

class Sequential:
    def __init__(self,layers:list[Layer]):
        self.layers = layers
        
    def predict(self,x):
       prev_a = x
       for layer in self.layers:
           a = layer(prev_a)
           prev_a = a

        return layer[-1].a_out
           
    
if __name__ == "__main__":
    l1 = Layer(units = 3)
    x_train = np.array([1.0,3.0])
    prediction = l1(x_train)
    print(prediction)
