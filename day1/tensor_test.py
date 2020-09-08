# tensorflow 2.0 basic
import tensorflow as tf

a = tf.constant([1,2,3])
print(a,'\n')
print(a.numpy())

b = tf.constant([1,2,3], dtype=tf.float32)
print('\n',b)

c = tf.range(5)
print('\n', c)

d = tf.linspace(0.,5., 11)
print('\n',d)
print('\n', tf.zeros([2,3]))
print('\n', tf.fill([2,3], 2)) # 2로 다채워라
print('\n', tf.zeros_like([[1,2,3], [4,5,6]])) # 2x3 형태의 0으로 채우기

tf.random.set_seed(111)
print('\n', tf.random.normal([2,3])) # 가우시안 값 랜덤
print('\n', tf.random.uniform([2,3])) # 0~1
print('\n', tf.random.shuffle([1,2,3,4,5])) # list shuffle

e = tf.constant([[2,4,7], [4,9,11]])
print('\n', e.ndim)
print('\n', e.shape)
print('\n', e.dtype)
print('\n', e.numpy())

'''
tf.reshape : 벡터행렬 연산 크기 변환
tf.transpose : 전치연산
tf.expand_dim : 지정축으로 차원추가
tf.squeeze: 벡터 차원축소
'''

f = tf.range(6, dtype=tf.int32)
print('\n',f)
rData = tf.reshape(f, (2,3))
print('\n',rData)

tData = tf.transpose(rData)
print('\n', tData)

eData = tf.expand_dims(rData, axis=0) # 축방향으로 차원추가
print('\n',eData)

sData = tf.squeeze(eData)
print('\n',sData)

g = tf.reshape(tf.range(12), [3,4])
print('\n', g)

g1, g2 = tf.slice(g, [0,1], [2,3]) # point 영역, 0,1 point 포함 부터 ~ 2,3 point 포함 하지 않는 data
print('\n', g1)
print('\n', g2)

g1, g2 = tf.split(g, num_or_size_splits=2, axis=1) # 축방향으로 2개 짜르겠다.
print('\n', g1)
print('\n', g2)
print('\n', tf.concat([g1, g2], axis=1))
print('\n', tf.stack([g1, g2]))
