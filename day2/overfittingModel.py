from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(3)

df = pd.read_csv('datasheet/wine.csv',header=None)

df = df.sample(frac=0.15) #15%만
data = df.values

x_data = data[:, 0:12]
y_data = data[:,12]

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=3)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
# overfitting 방지 조기 멈춤
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

model_dir ='./model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

estopcallback = EarlyStopping(monitor='val_loss', patience=100) # validation loss 값 계속 확인해서, 100개까지 안좋아지는 순간이 발생하면 종료한다. 100단위로 확인
savecallback = ModelCheckpoint(filepath='./model/{val_loss:.4f}.hdf5',monitor='val_loss', save_best_only=True) # model 자체 save가 아니라 weight만 저장하는 function


# data의 33%는 validation으로 쓰겠다.
result = model.fit(x_data,y_data, validation_split=0.33, epochs=3500, batch_size=500, callbacks=[estopcallback, savecallback])
# result = model.fit(x_train,y_train, validation_split=0.33, epochs=3500, batch_size=500, callbacks=[estopcallback, savecallback])
# overfitting 발생

y_validation_loss = result.history['val_loss']
y_acc = result.history['accuracy']

import numpy as np

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_validation_loss, 'o', c='red', ms=2, label='validation_loss')
plt.plot(x_len, y_acc, 'o', c='blue', ms=2, label='accuracy')
plt.legend(loc='best')
plt.show()
'''

model.load_weights('./model/0.0536.hdf5')
print(model.evaluate(x_data,y_data))

# print(model.evaluate(x_test,y_test))
