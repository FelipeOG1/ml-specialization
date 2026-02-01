import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCroseentropy


class Model:
    def __init__(self):
        self._model = Sequential([
            Dense(units = 25, activation = 'sigmoid'),
            Dense(units = 15, activation = 'sigmoid'),
            Dense(units = 1,activation= 'sigmoid')
        ])

    def __getitem__(self,pos:int):
        return self._model[pos]
    
    def train(self,x,y):
        self._model.compile(loss=BinaryCroseentropy)
        self._model.fit(x,y,epochs=100)
        



