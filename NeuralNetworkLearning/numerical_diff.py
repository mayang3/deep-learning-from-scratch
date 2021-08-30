# 나쁜 구현의 예
# def numerical_diff(f, x):
#     h = 10e-50
#     return (f(x + h) - f(x)) / h


# 개선된 구현
# 1. h 값으로 미세한 값으로 반올림 오차를 유발하는 값 대신에 적절한 값을 할당한다.
# 2. 함수 f 의 차분에 관련된 것이다. 진정한 미분은 극한의 개념이기 때문에 x 위치의 함수의 기울기(접선) 에 해당하지만, 이번 구현에서의 미분은 (x+h) 와 x 사이의 기울기에 해당한다.
# 그래서 진정한 미분과 이번 구현의 값은 엄밀히는 일치하지 않는다. 이 차이는 h 를 무한히 0 으로 좁히는 것이 불가능해 생기는 한계이다.
# 이 오차를 줄이기 위해 (x+h) 와 (x-h) 일 때, 함수 f 의 차분을 계산하는 방법을 쓰기도 한다. 이 차분은 x 를 중심으로 그 전후의 차분을 계산한다는 의미에서 중심 차분 혹은 중앙 차분이라고 한다.
# 한편, (x+h) 와 x 의 차분은 전방 차분이라고 한다.
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)  # 0~20 까지 0.1 간격의 배열 x 를 만든다.
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))


# 편미분 예제 구현
def function_2(x):
    # 또는 return np.sum(x**2)
    return x[0] ** 2 + x[1] ** 2


# 편미분 문제1)
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0


print(numerical_diff(function_tmp1, 3.0)) # 6.00000000000378
# 해석적 미분의 결과와 거의 같다.

