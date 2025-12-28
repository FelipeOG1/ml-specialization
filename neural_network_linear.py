
from decimal import DecimalTuple
import numpy as np
import tensorflow as tf


    
    


class NeuralNetwork:
    def __init__(self,X,y_train):
        self.X = X
        self.y_train = y_train
        self.linear_layer = tf.keras.layers.Dense(units = 1,activation = 'linear')
        self.linear_layer.build(input_shape = (None,1))#need this for custom weights
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units = 1,input_dim = 1,activation = 'sigmoid',name = "L1")
            ])
    @property
    def weights(self):
        kernel_w,kernel_b = self.model.weights
        return kernel_w.numpy(),kernel_b.numpy()
        
    def setup_custom_weights(self,w:float,b:float)->None:
        set_w = np.array([[w]],dtype = np.float32)
        set_b = np.array([b],dtype = np.float32)
        self.linear_layer.set_weights([set_w,set_b])
        self.model.get_layer('L1').set_weights([set_w,set_b])
        assert(self.linear_layer.get_weights() == [[set_w],[set_b]])
        
    def test_linear_layer_activation(self):
        w,b = self.linear_layer.get_weights()
        a1 = self.linear_layer(self.X[0].reshape(1,1))
        manual_a1 = np.dot(self.X[0].reshape(1,1),w) + b
        assert(a1 == manual_a1)
    def test_sigmoid_activation(self):
        w,b = self.weights
        sigmoid = lambda z:1/(1+np.exp(-z))
        a1 = self.model.predict(self.X[0].reshape(1,1))
        alog = sigmoid(np.dot(self.X[0].reshape(1,1),w) + b)
        assert(a1 == alog)
        
x =  np.array([[1.0], [2.0]], dtype=np.float32)#a0
y = np.array([[300.0], [500.0]], dtype=np.float32)
nn = NeuralNetwork(x,y)

w,b = 2.0,-4.5
nn.setup_custom_weights(w,b)
print(nn.test_sigmoid_activation())

