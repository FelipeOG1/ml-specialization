from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import numpy as np

model1 = Sequential([
            Dense(units = 25,activation = 'relu'),
            Dense(units = 15,activation = 'relu'),
            Dense(units = 10,activation ='softmax')
        ])


model2 = model1 = Sequential([
            Dense(units = 25,activation = 'relu'),
            Dense(units = 15,activation = 'relu'),
            Dense(units = 10,activation ='linear')
        ])

shape = (20,400)
classes =  10
x_train = np.random.randn(*shape)
y_train = np.random.randint((classes - classes),classes,size=(x_train.shape[0],1))


#LOWER PRECISSION
model1.compile(loss=SparseCategoricalCrossentropy)
model1.fit(x_train,
           y_train,
           epochs=100
           )

prediction = model1(x_train)



#HIGHER PRECISSION
model2.compile(SparseCategoricalCrossentropy(from_logits=True))
model2.fit(x_train,
           y_train,
           epochs=100
           )

logits  = model2(x_train)
prediction = tf.nn.softmax(logits)




