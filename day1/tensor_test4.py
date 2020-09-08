import tensorflow as tf

def f(x):
    tv = tf.Variable([[4,5], [9,10]])
    return tv * x


print(f(10))

@tf.function # 컴파일시 조금더 유리 하다. 속도면에서

def f2(a,b):
    return tf.matmul(a,b)

x = [[4,5,6], [7,8,9]]
w = tf.Variable([[2,4], [7,5], [9,10]])
print(w)
print('\n', f2(x,w))
