import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 2 version 에서 1version 사용하기
'''
a = tf.constant(5)
b = tf.constant(10)
c = a*b
# print(c)
sess = tf.Session()
result = sess.run(c)
print('\n', result)
print('\n', sess.run([a,b,c]))


input_data = [2,6,3,8,9]
x = tf.placeholder(dtype=tf.float32)
y = x * 2

sess = tf.Session()
result2 = sess.run(y,feed_dict={x : input_data})
print(result2)

'''

import numpy as np
np.random.seed(12345)
x_data = np.random.randn(5,10)
w_data = np.random.randn(10, 1)

x = tf.placeholder(dtype=tf.float32, shape=[5,10])
w = tf.placeholder(dtype=tf.float32, shape=[10,1])
b = tf.fill([5,1], -1.)

output = tf.matmul(x, w) + b
result3 = tf.reduce_max(output) # axis 가 none 이면 모든 dimension에서 진행
sess = tf.Session()
print('\n', sess.run(result3, feed_dict={x: x_data, w:w_data}))

