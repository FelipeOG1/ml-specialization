import numpy as np
from dataclasses import dataclass


def g(z):return 1/(1+np.exp(-z))

@dataclass
class Layer:
    units:int
    w:np.ndarray | None = None
    b:np.ndarray | None = None
      
class Sequential:
    def __init__(self,layers:list[Layer]):
        self.layers = layers

    
    def predict(self,x:np.ndarray):
        a = x
        for layer in self.layers:
            a = g((a @ layer.w) + layer.b)
        return a

    def __getitem__(self,position:int):
        return self.layers[position]
    
    def set_weights(self,weights:list[np.ndarray]):
        assert len(weights) == 2 * len(self.layers)
        for index,layer in enumerate(self.layers):
            layer.w,layer.b = weights[index * 2],weights[index * 2 + 1]


        

