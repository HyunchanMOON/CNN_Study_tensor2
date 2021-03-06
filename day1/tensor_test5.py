import tensorflow as tf

x = tf.Variable(tf.constant(1.0))
print(x)

with tf.GradientTape() as tape:
    y = tf.multiply(5, x)

gradient = tape.gradient(y, x)
print(gradient)

x1 = tf.Variable(tf.constant(1.0))
x2 = tf.Variable(tf.constant(2.0))

with tf.GradientTape() as tape:
    y = tf.multiply(x1, x2)

print(y)
gradient = tape.gradient(y, [x1, x2]) # 편미분

print(gradient)
print(gradient[0].numpy())
print(gradient[1].numpy())

x3 = tf.Variable(tf.constant(1.0))
a = tf.constant(2.)

with tf.GradientTape() as tape:
    tape.watch(a) # 상수를 변수로 변환
    y = tf.multiply(a, x3)

gradient2 = tape.gradient(y, x3)

print('\n', gradient2)
