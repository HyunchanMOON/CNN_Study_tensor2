import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.set_random_seed(777)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


#iris 4가지 정보 값
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,7,6,6],
          [1,7,7,7]]

# one hot encoding이 된 결과값
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

## weight sum 만들기
w1 = tf.Variable(tf.random_normal([4,3]))
b1 = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(x,w1)+b1)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1)) # 출력 3x1 형태로 출력됨  그렇기에 열방향 합
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(2000):
        _, _c = sess.run([train, cost], feed_dict={x:x_data, y:y_data})
        if epoch % 100 == 0:
            print('epoch:{} cost:{:.4f}'.format(epoch, _c))


    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print('\nhypothesis:{}\nprediction:{}'.format(a, sess.run(tf.argmax(a, axis=1))))

    b = sess.run(hypothesis, feed_dict={x:[[1,11,7,9],
                                           [1,3,4,2],
                                           [1,1,0,1]]})
    print('\nhypothesis:\n{}\nprediction:\n{}'.format(b, sess.run(tf.argmax(b, axis=1))))
