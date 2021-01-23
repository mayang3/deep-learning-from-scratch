import numpy as np
import matplotlib.pylab as plt


# 가장 단순한 계단함수
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


print(step_function(3.0))


# 넘파이 배열도 지원하는 계단함수
def step_function(x):
    y = x > 0  # 넘파이의 부등호 연산을 수행
    return y.astype(np.int)  # Bool 배열을 int 형으로 변환


print(step_function(np.array([1.0, 2.0])))


# 계단 함수의 그래프
def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)  # -5.0 ~ 5.0 까지의 범위로 0.1 간격의 넘파이 배열을 생성한다.
y = step_function(x)  # 인수로 받은 넘파이 배열의 원소 각각을 인수로 계단 함수를 실행해, 그 결과를 다시 배열로 만들어 돌려준다.
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

