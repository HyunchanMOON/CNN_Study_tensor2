# mnist 이미지 데이터를 이용해 CNN 모델 구현하기
# CNN 모델을 클래스(Sequential 상속)으로 구현하기
# 학습이 끝난 후 모델을 평가하여 loss와 accuracy를 출력하기

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def Data_func():
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    print(x_train.shape)
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


class CNN_seq_class(Sequential):
    def __init__(self, in_sh, ker_size, filter, Nh, Nout):
        super().__init__()
        self.add(Conv2D(input_shape=in_sh, kernel_size=ker_size, filters=filter, padding='same', activation='relu'))
        self.add(Conv2D(kernel_size=ker_size, filters=filter*2, padding='same', activation='relu'))
        self.add(MaxPool2D(strides=(2,2)))
        self.add(Dropout(rate=0.5))
        self.add(Conv2D(kernel_size=ker_size, filters=filter*4, padding='same', activation='relu'))
        self.add(Conv2D(kernel_size=ker_size, filters=filter*8, padding='same', activation='relu'))
        self.add(MaxPool2D(strides=(2,2)))
        self.add(Dropout(rate=0.5))
        self.add(Flatten())
        self.add(Dense(units=Nh, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(Dense(units=Nh/2, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(Dense(Nout, activation='relu'))
        self.compile(loss='sparse_categorical_crossentropy', # keras.losses.categorical_crossentropy
                    optimizer=tf.keras.optimizers.Adam(lr=0.001), # rms
                    metrics=['accuracy'])

(x_train, y_train),(x_test, y_test) = Data_func()

input_shape = x_train.shape
in_sh = input_shape[1:]
print(in_sh)
ker_size = (3,3)
filter = 32
Nh = 512
numberofClass = 10
Nout = numberofClass
model = CNN_seq_class(in_sh, ker_size, filter, Nh, Nout)
model.summary()
# result = model.fit(x_train, y_train, epochs=25, validation_split=0.25, batch_size=100)
# print('loss & accuracy:',model.evaluate(x_test, y_test, batch_size=100))
