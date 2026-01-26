import numpy as np


class Neuron:
    def compute_scalar(self,x,w,b):
        sigmoid = lambda z:1/(1+np.exp(-z))
        return sigmoid(-(np.dot(x,w)) + b)    

class Layer:
    def __init__(self,units:int):
        self._core = [Neuron() for _ in range(units)]
        self.units = units
        self.a_out = np.zeros(units)

    def __call__(self,a_in,w,b):#returns a_out
        for index,neuron in enumerate(self._core):
            curr_w = w[:,index]
            curr_b = b[index]
            res = neuron()


    def __len__(self):
        return len(self._core)
    
    def __getitem__(self,position:int):
        if position>=len(self._core):raise ValueError(f"Out of bounce,this layer only have {self.units}units")
        return self._core[position]

    





in_layer = np.array([-2,4])

w = np.array([[1,-3,5],
              [2,4,6]])

b = np.array([-1,1,2])

l1 = Layer(units=3)
l2 = Layer(units=1)


a1_out = l1(a_in=in_layer,
            w=w,
            b=b
            )


print(a1_out)   
