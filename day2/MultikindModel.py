from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input

def Data_func():
    #tuple 형태로
    # [batch, 28 x 28 = 784 ] 형태와 255로 나눈값
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def NN_model_func(Nin, Nh, Nout):
    model = Sequential()
    model.add(Dense(Nh, activation='relu', input_shape=(Nin,)))
    model.add(Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# 좀더 low 한 coding
def NN_model_func2(Nin, Nh, Nout):
    x = Input(shape=(Nin,)) # input 받고
    h = Activation('relu')(Dense(Nh)(x)) # 그다음 layer로 값ㅇ르 넘겨준다.
    y = Activation('softmax')(Dense(Nout)(h)) # 그다음 activate fun 으로 넘긴다.
    model = Model(x,y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class NN_seq_class(Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# 가장 많이 사용됨.
class NN_model_class(Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = Dense(Nh)
        output = Dense(Nout)
        relu = Activation('relu')
        softmax = Activation('softmax')

        x = Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))
        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])



Nin = 784
Nh = 100
numberofClass = 10
Nout = numberofClass

# model = NN_model_func(Nin, Nh, Nout)
# model = NN_model_func2(Nin, Nh, Nout)
# model = NN_seq_class(Nin, Nh, Nout)
model = NN_model_class(Nin, Nh, Nout)

(x_train, y_train),(x_test, y_test) = Data_func()
model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=100)

print('loss & accuracy:',model.evaluate(x_test, y_test, batch_size=100))



