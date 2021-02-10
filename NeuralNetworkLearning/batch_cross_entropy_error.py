import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from NeuralNetwork.neuralnet_mnist_batch import init_network, predict

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784) -> 훈련 데이터 60000 개, 입력 데이터 784 열
print(t_train.shape)  # (60000, 10) -> 정답 레이블 60000 개, 정답 데이터 10 열

train_size = x_train.shape[0]
batch_size = 10
# 0~60000 사이의 수 중 무작위로 10개를 골라낸다. (훈련 데이터 중 10개를 무작위로 골라내는 의미이다.)
batch_mask = np.random.choice(train_size, batch_size)

print(batch_mask)  # [19983  5895 16784 16337 26519  1102 43928 44010 44903 57550] -> random array

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# get y_batch
network = init_network()
y_batch = predict(network, x_batch)


def cross_entropy_error(y, t):
    #  1 차원 배열인 경우, 배열의 형상을 바꿔준다.
    # 예를 들어, t = [0,0,1,0,0] 이라면 t.reshape(1,1) 을 실행하면 t=[[0,0,1,0,0]] 과 같은 2차원 배열이 된다.
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


print(cross_entropy_error(y_batch, t_batch))  # 3.1737068235874175
