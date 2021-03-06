import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)
learing_rate = 0.1

x_data = [[0,0],[0,1], [1,0], [1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])


#layer 1
w1 = tf.Variable(tf.random_normal([2,10]))
b1 = tf.Variable(tf.random_normal([10]))
layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)

w2 = tf.Variable(tf.random_normal([10,10]))
b2 = tf.Variable(tf.random_normal([10]))
layer2 = tf.sigmoid(tf.matmul(layer1,w2)+b2)

w3 = tf.Variable(tf.random_normal([10,10]))
b3 = tf.Variable(tf.random_normal([10]))
layer3 = tf.sigmoid(tf.matmul(layer2,w3)+b3)

w4 = tf.Variable(tf.random_normal([10,1]))
b4 = tf.Variable(tf.random_normal([1]))

'''
#1 layer perceptron
w1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.random_normal([2]))
layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)

w2 = tf.Variable(tf.random_normal([2,1]))
b2 = tf.Variable(tf.random_normal([1]))
'''

hypothesis = tf.sigmoid(tf.matmul(layer3, w4) + b4)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

update = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 크면 1 작으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32)) # casting 한다. 같으면 1 다르면 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        _cost, _ = sess.run([cost, update], feed_dict={x:x_data, y:y_data})
        if epoch % 200 == 0:
            print('epoch:{} cost:{}'.format(epoch, _cost))
    _h, _p, _a = sess.run([hypothesis, prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print('\nhypothesis:\n{}\nprediction:\n{}\naccuracy:{}'.format(_h, _p, _a))
