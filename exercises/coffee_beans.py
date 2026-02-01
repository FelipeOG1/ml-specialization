import numpy as np
import tensorflow as tf
class BinaryPredictor:
    def __init__(self,x,y):
        self.X = x
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units = 3,activation = 'sigmoid'),
            tf.keras.layers.Dense(units = 1 ,activation = 'sigmoid')])
        
            
   
