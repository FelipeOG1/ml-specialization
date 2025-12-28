


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
class NeuralNetwork:
    def __init__(self,X,y_train):
        self.X = X
        self.y_train = y_train
        self.layer = tf.keras.layers.Dense(units = 1,activation = 'linear')
        self.a1 = self.layer(self.X[0].reshape(1,1))
    @property
    def weights(self):return self.layer.get_weights()
    

    


x =  np.array([[1.0], [2.0]], dtype=np.float32)#a0
y = np.array([[300.0], [500.0]], dtype=np.float32)
nn = NeuralNetwork(x,y)


w,b = nn.weights
print(f"w{w}")
print(f"b{b}")
