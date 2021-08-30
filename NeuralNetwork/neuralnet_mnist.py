import sys, os

sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

# 학습 과정은 생략하고, 순전파 과정만 구현한다.

def get_date():
    # load_mnist 함수로 MNIST 데이터셋을 읽는다.
    # 읽혀진 데이터를 (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 변환한다.

    # normalize : 입력 이미지의 픽셀값을 0.0~1.0 사이의 값으로 정규화할지를 결정한다. False 로 설정하면 원래 값 그대로 0~255 사이의 값을 유지한다.
    # flatten : 입력 이미지를 평탄하게, 즉 1차원 배열로 만들것인지를 정한다.
    # one_hot_label : 레이블을 원-핫-인코딩 형태로 저장할지를 결정한다.
    # 원-핫 인코딩이란, 예를 들어 {0,0,1,0,0,0,0} 처럼 정답을 뜻하는 원소만 1이고(hot) 나머지는 모두 0인 배열이다.
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


# 입력층 뉴런 784개 (이미지 크기 28 * 28 = 784 이기 때문)
# 출력층 뉴런 10 개 (문제가 0~9까지의 숫자를 구분하는 문제이기 때문)
# 은닉층은 총 두개로 첫 번째 은닉층에는 뉴런 50개, 두 번째 은닉층에는 뉴런 100개 (임의값)
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
    # x[i] 는 784 짜리 길이의 하나의 row 인 행렬이다. (1 * 784)
    y = predict(network, x[i])  # 각 레이블의 확률을 넘파이 배열로 반환한다.
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:  # 찾은 값을 정답 레이블과 비교한다.
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
