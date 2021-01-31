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

print(x.shape)  # (10000, 784) -> 1차원 배열 784 로 표현된 1개의 이미지가 10000개가 있다.
print(x[0].shape)  # (784,)

# 학습된 가중치 매개변수를 가지고 있는 신경망을 가져온다.
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0
# 각 이미지마다 추론을 시작한다.
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # 1번째 차원을 구성하는 각 원소에서 최대값의 인덱스를 찾게 해줌
    # 여기서 1번째 차원은 2차원이기 때문에, 2차원을 구성하는 각 열마다의 최대값을 구해줌(즉 배열 하나 안에서의 최대값을 구해줌)
    p = np.argmax(y_batch, axis=1)
    # 넘파이 배열끼리 비교하여 True/False bool 배열을 만들고, 이 결과 배열에서 True 가 몇개인지를 센다
    accuracy_cnt += np.sum(p == t[i:i+batch_size])


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352