import numpy as np


def softmax(a):
    c = np.max(a)  # 오버플로 방지
    exp_a = np.exp(a)  # 지수 함수
    sum_exp_a = np.sum(exp_a)  # 지수함수의 합
    y = exp_a / sum_exp_a
    return y


# softmax 함수를 사용한 신경망 출력의 예
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)  # [0.01821127 0.24519181 0.73659691]
print(np.sum(y))  # 1.0
