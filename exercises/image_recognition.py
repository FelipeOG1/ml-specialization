import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
   
class Model:
    def __init__(self):
        self._core = Sequential([
            Dense(units = 25, activation = 'sigmoid'),
            Dense(units = 15, activation = 'sigmoid'),
            Dense(units = 1,activation= 'sigmoid')
        ])

    def __getitem__(self,pos:int):
        return self._core[pos]



