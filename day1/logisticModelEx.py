import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,2],
          [6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])# none은 앞에 들어오는건 상관없이 쓴다. batch size 맞춰주기 위해서.
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
w = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # x1w1 + x2w2 + b -> # 1/1+e^-(ax+b)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # cross entropy
update = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
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
