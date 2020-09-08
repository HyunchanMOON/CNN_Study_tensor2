# http://conlms.lge.com/dwn/ext/200/G006/2020/09/01/d60e6c73-7edf-4bbb-a18c-a63e8c61c3f3.zip
# VGG 16

import tensorflow as tf
import matplotlib.pyplot as plt

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

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# y : 0~9 , onehot 진행 안했다!
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(-1, 28,28, 1) # 4 tensor 형태, [batch, H, W, Channel]
x_test = x_test.reshape(-1, 28,28, 1)
input = x_train.shape
input = input[1:]
print(type(input))
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

x_train = x_train / 255.
x_test = x_test / 255.

model = tf.keras.Sequential([
    Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, padding='same', activation='relu'),
    Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    MaxPool2D(strides=(2,2)),
    Dropout(rate=0.5),
    Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
    Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
    MaxPool2D(strides=(2,2)),
    Dropout(rate=0.5),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dropout(rate=0.5), # 50 % 를 줄이겠다. 랜덤하게, batch 마다 randomly
    Dense(units=256, activation='relu'),
    Dropout(rate=0.5), # 50 % 를 줄이겠다. 랜덤하게
    Dense(10, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
#sparse는 정수로 들어오는 값,

model.summary()

# result = model.fit(x_train, y_train, epochs=3, validation_split=0.25, batch_size=100)
# print(model.evaluate(x_test, y_test))
#
# plt.figure(12,4)
# plt.subplot(1,2,1)
# plt.plot(result.history['loss'], 'b-', label='loss')
# plt.plot(result.history['val_loss'], 'r-', label='val_loss')
# plt.xlabel('epoch')
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.plot(result.history['accuracy'], 'b-', label='accuracy')
# plt.plot(result.history['val_accuracy'], 'r-', label='val_accuracy')
# plt.xlabel('epoch')
# plt.ylim(0.7, 1)
# plt.legend()
#
# plt.show()
