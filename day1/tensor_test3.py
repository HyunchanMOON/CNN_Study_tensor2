import tensorflow as tf

vdata1 = tf.Variable(1)
print(vdata1)
vdata2 = tf.Variable(tf.ones(2,))
print('\n',vdata2)
vdata3 = tf.Variable(tf.ones([2,3]))
print('\n',vdata3)
vdata3.assign((tf.zeros([2,3])))
print('\n',vdata3)
print('\n', vdata3.assign_add(tf.ones([2,3])))
