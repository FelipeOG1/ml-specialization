import numpy as np

def g(z):return 1/(1+np.exp(-z))

class Neuron:
    def compute_scalar(self,x,w,b):
        sigmoid = lambda z:1/(1+np.exp(-z))
        return sigmoid((np.dot(x,w)) + b)

class Layer:
    def __init__(self,units:int):
        self._core = [Neuron() for _ in range(units)]
        self.units = units
        self.a_out = np.zeros(units)


    def vectorized_a_out(self,a_in,w,b):return g(np.matmul(a_in,w) + b)
        
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
            
        weight_index = 0
        current_input = x_train
        
        for layer in self.layers:
            w = self._weights[weight_index]
            b = self._weights[weight_index + 1]
            current_input = layer(current_input,w,b)
        
            weight_index+=2

        return current_input

    def __len__(self):
        return len(self.layers)
            
        
    def set_weights(self,weights:list[np.ndarray]):
        if len(weights) != len(self.layers) * 2:raise ValueError("Wrong number of weights")
        self._weights = weights
     
 jk        
        
    
