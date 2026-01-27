import numpy as np


class Neuron:
    def compute_scalar(self,x,w,b):
        sigmoid = lambda z:1/(1+np.exp(-z))
        return sigmoid((np.dot(x,w)) + b)    

class Layer:
    def __init__(self,units:int):
        self._core = [Neuron() for _ in range(units)]
        self.units = units
        self.a_out = np.zeros(units)

    def __call__(self,a_in,w,b):#returns a_out
        for index,neuron in enumerate(self._core):
            curr_w = w[:,index]
            curr_b = b[index]
            self.a_out[index] = neuron.compute_scalar(x=a_in,
                                                      w=curr_w,
                                                      b=curr_b
                                                      )
        return self.a_out
        

    def __len__(self):
        return len(self._core)
    
    def __getitem__(self,position:int):
        if position>=len(self._core):raise ValueError(f"Out of bounce,this layer only have {self.units}units")
        return self._core[position]



class Sequential:
    
    def __init__(self,layers:list[Layer]):
        self.layers = layers
        self._weights = None
    def predict(self,x_train:np.ndarray):
        if not self._weights:raise ValueError("Add weithgs by set_weights")
        for index,layer in enumerate(self.layers):
            a_in = x_train
            weights = self._weights[:index + 2]
            
        
    def set_weights(self,weights:list[np.ndarray]):
        if len(weights) != len(self.layers) * 2:raise ValueError("Wrong number of weights")
        self._weights = weights
        
    
    

in_layer = np.array([-2,4])

w = np.array([[1,-3,5],
              [2,4,6]])

b = np.array([-1,1,2])

w2 = np.array([[5,-1,2],
              [1,3,10]])


b2 = np.array([-3,5,1])


custom_weights = [w,b,w2,b2]

model = Sequential([
    Layer(units=3),
    Layer(units=1)
])




model.set_weights(custom_weights)


