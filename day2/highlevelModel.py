import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

np.random.seed(3)
tf.random.set_seed(3)

data = np.loadtxt('datasheet/ThoraricSurgery.csv', delimiter=',')
x_data = data[:, 0:17]
y_data = data[:, 17]

#Sequential 이용해서 model 객체 생성
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu')) #layer 1, w와 b는 저절로 생성됨.
model.add(Dense(1, activation='sigmoid')) # 출력층

## 손실함수, 최적화 함수, 설정
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#feed 값 적용하는 과정
result = model.fit(x_data, y_data, epochs=100, batch_size=10)

print(result.history['loss'])
print('\n',result.history['accuracy'])



