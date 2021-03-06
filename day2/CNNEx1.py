# http://conlms.lge.com/dwn/ext/200/G006/2020/09/01/d60e6c73-7edf-4bbb-a18c-a63e8c61c3f3.zip

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
# plt.figure(figsize=(10,10))
# for c in range(16):
#     plt.subplot(4,4, c+1)
#     plt.imshow(x_train[c].reshape(28,28), cmap='gray')
# plt.show()

'''
0:티셔츠
1:바지
2:스웨터
3:드레스
4:코트
5:샌들
6:셔츠
7:운동화
8:가방
9:부츠
'''
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

x_train = x_train / 255.
x_test = x_test / 255.

model = tf.keras.Sequential([
    Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, activation='relu'),
    MaxPool2D(strides=(2,2)),
    Conv2D(kernel_size=(3,3), filters=64, activation='relu'),
    MaxPool2D(strides=(2,2)),
    Conv2D(kernel_size=(3,3), filters=128,activation='relu'),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(rate=0.3), # 30 % 를 줄이겠다. 랜덤하게
    Dense(10, activation='softmax')
])

# model = tf.keras.Sequential([
#     Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32,padding='same', activation='relu'),
#     Conv2D(kernel_size=(3,3),filters=64, padding='same',activation='relu'),
#     Conv2D(kernel_size=(3,3), filters=128,padding='same',activation='relu'),
#     Flatten(),
#     Dense(units=128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
#sparse는 정수로 들어오는 값,

model.summary()

result = model.fit(x_train, y_train, epochs=5, validation_split=0.25, batch_size=100)
print(model.evaluate(x_test, y_test))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(result.history['loss'], 'b-', label='loss')
plt.plot(result.history['val_loss'], 'r-', label='val_loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(result.history['accuracy'], 'b-', label='accuracy')
plt.plot(result.history['val_accuracy'], 'r-', label='val_accuracy')
plt.xlabel('epoch')
plt.ylim(0.7, 1)
plt.legend()
plt.show()
