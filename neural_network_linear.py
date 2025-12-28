


from decimal import DecimalTuple
import numpy as np
from numpy._core.numeric import dtype
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.python.ops.gen_experimental_dataset_ops import weighted_flat_map_dataset
class NeuralNetwork:
    def __init__(self,X,y_train):
        self.X = X
        self.y_train = y_train
        self.layer = tf.keras.layers.Dense(units = 1,activation = 'linear')
        self.a1 = self.layer(self.X[0].reshape(1,1))
        
    @property
    def weights(self):return self.layer.get_weights()
    
    def setup_custom_weights(self,w:float,b:float)->None:
        set_w = np.array([[w]],dtype = np.float32)
        set_b = np.array([b],dtype = np.float32)
        self.layer.set_weights([set_w,set_b])
        assert(self.layer.get_weights() == [[set_w],[set_b]])
        


x =  np.array([[1.0], [2.0]], dtype=np.float32)#a0
y = np.array([[300.0], [500.0]], dtype=np.float32)
nn = NeuralNetwork(x,y)

w,b = 200.0,100.0
nn.setup_custom_weights(w,b)
print(nn.weights)
