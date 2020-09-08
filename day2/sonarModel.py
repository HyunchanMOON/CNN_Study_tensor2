from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # 문자열 인코딩
import pandas as pd
import numpy as np
import tensorflow as tf

tf.random.set_seed(3)

df = pd.read_csv('datasheet/sonar.csv',header=None)
data = df.values # data frame에서 data만 뽑기
x_data = data[:, 0:60].astype(float)
y_data = data[:,60] # R,M 되어있는 값 바꿔줘야한다.
le = LabelEncoder()
le.fit(y_data) # uniquce 한 값 뽑아서 인코딩을 진행한다.
y_data = le.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=3)

#Sequential 이용해서 model 객체 생성
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu',name='layer1')) #layer 1, w와 b는 저절로 생성됨.
model.add(Dense(10, activation='relu',name='layer2')) #layer 2
model.add(Dense(1, activation='sigmoid',name='layer3')) # 출력층

## 손실함수, 최적화 함수, 설정
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
#feed 값 적용하는 과정

result = model.fit(x_train, y_train, epochs=200, batch_size=5)
model.save('smodel.h5') # model 자체를 save하는 과정

del model

from tensorflow.keras.models import load_model
model = load_model('smodel.h5')

e_result = model.evaluate(x_test,y_test)

print(e_result) # loss, accuracy

