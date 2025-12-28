import numpy as np
import tensorflow as tf
class BinaryPredictor:
    def __init__(self,x,y):
        self.X = x
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units = 3,activation = 'sigmoid'),
            tf.keras.layers.Dense(units = 1 ,activation = 'sigmoid')])
            

        
    def get_predictions(self,x_new):
        """
        self.model.compile()
        self.model.fit()
        return self.model.predict()
        """
    
x = np.array([[200.0,17.0],
              [120.0,5.0],
              [425.0,20.0],
              [212.0,18.0]])

y = np.array([1,0,0,1])

bn = BinaryPredictor(x,y)
print(bn.model.summary())



