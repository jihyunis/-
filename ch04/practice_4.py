import numpy as np
import matplotlib.pyplot as plt

# 평균 제곱 오차 mean squared error
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# print(mean_squared_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))

# 교차 엔트로피 오차 cross entropy error
x = np.arange(0.001, 1.0, 0.001)
y = np.log(x)

# plt.plot(x, y)
# plt.ylim(-5.0, 0.0) # y축의 범위 지정
# plt.show()

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # 아주 작은 값 delta를 더하는 이유 : np.log에 함수 0을 입력하면 -inf(무한)가 되어 계산 x

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(cross_entropy_error(np.array(y), np.array(t)))

# 미니 배치 학습

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# 훈련 데이터에서 무작위로 10장만 꺼내기? np.random.choice() 사용
# np.random.choice(60000,10)은 0에서 60000 미만의 수 중 무작위로 10개를 골라냄
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# (배치용) 교차 엔트로피 오차 구현하기
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

