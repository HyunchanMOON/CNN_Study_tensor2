import numpy as np
t  = [0,   0,    1,   0,   0,    0,   0,   0,   0,   0] # 정답 label
y1 = [0.1, 0.05, 0.7, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0] # softmax 출력값
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # softmax 출력값 오류
### 10개의 출력 노드 ###

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# MSE 와 cross entropy 비교
mse1 = mean_squared_error(np.array(y1), np.array(t))
print('mse1:',mse1)
mse2 = mean_squared_error(np.array(y2), np.array(t))
print('mse1:',mse2)

cee1 = cross_entropy_error(np.array(y1), np.array(t))
print('\ncee1:',cee1)
cee2 = cross_entropy_error(np.array(y2), np.array(t))
print('\ncee1:',cee2)
