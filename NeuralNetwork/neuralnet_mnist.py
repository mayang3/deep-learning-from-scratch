import sys, os

sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist


def get_date():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# 학습된 가중치/편향의 매개변수를 읽는다.
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)  # 오버플로 방지
    exp_a = np.exp(a)  # 지수 함수
    sum_exp_a = np.sum(exp_a)  # 지수함수의 합
    y = exp_a / sum_exp_a
    return y


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 이미지 데이터를 가져온다.
x, t = get_date()

# 학습된 가중치 매개변수를 가지고 있는 신경망을 가져온다.
network = init_network()

accuracy_cnt = 0
# 각 이미지마다 추론을 시작한다.
for i in range(len(x)):
    y = predict(network, x[i])  # 각 레이블의 확률을 넘파이 배열로 반환한다.
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:  # 찾은 값을 정답 레이블과 비교한다.
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
