import numpy as np
from dataclasses import dataclass


@dataclass
class Layer:
    units:int
    w:np.ndarray | None = None
    b:np.ndarray | None = None
    
    
class Sequential:
    def __init__(self,layers:list[Layer]):
        self.layers = layers

    
    def predict(self):
        pass

    def __getitem__(self,position:int):
        return self.layers[position]
    
    def set_weights(self,weights:list[np.ndarray]):
        assert len(weights) == 2 * len(self.layers)
        for index,layer in enumerate(self.layers):
            layer.w,layer.b = weights[index * 2],weights[index * 2 + 1]

model = Sequential([
    Layer(units=25),
    Layer(units=10)
])

a_in = np.random.rand(1000,400)

w1,b1 = np.random.rand(400,25),np.random.rand(25,)

w2,b2 = np.random.rand(25,10),np.random.rand(10,)

model.set_weights([w1,b1,w2,b2])

