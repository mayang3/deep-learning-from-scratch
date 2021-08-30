import numpy as np


# 편미분 예제 구현
def function_2(x):
    # 또는 return np.sum(x**2)
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))  # [6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0])))  # [0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))  # [6. 0.]

