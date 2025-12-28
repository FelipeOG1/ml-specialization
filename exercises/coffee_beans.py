




import numpy as np
import tensorflow as tf
class Model:
    
    def __init__(self):
        self.layer_1 = tf.keras.layers.Dense(units = 3,activation = 'sigmoid')
        self.layer_2 = tf.keras.layers.Dense(units = 1,activation = 'sigmoid')
    
        self.X = np.array([[200.0,17.0]])
       
    def get_prediction(self):return 1 if self.layer_2(self.layer_1(self.X))>=5 else 0



m = Model()
print(m.get_prediction())
