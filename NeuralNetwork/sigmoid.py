import numpy as np
import matplotlib.pylab as plt


# np.exp(-x) 가 넘파이 배열을 반환하기 때문에 1 / (1 + np.exp(-x)) 도 넘파이 배열의 각 원소에 연산을 수행한 결과를 내어준다.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 넘파이 배열도 처리 가능
print(sigmoid(np.array([-1.0, 1.0, 2.0])))

"""
위의 함수가 넘파이 배열도 훌륭히 처리해줄 수 있는 비밀은 넘파이의 브로드캐스트에 있다.

브로드캐스트 기능이란 넘파이 배열과 스칼라값의 연산을 넘파이 배열의 원소 각각과 스칼라값의 연산으로 바꿔 수행하는 것이다.
"""

# 넘파이 브로드캐스트의 예시

t = np.array([1.0, 2.0, 3.0])

print(1.0 + t)  # [2. 3. 4.]
print(1.0 / t)  # [1.         0.5        0.33333333]


# 시그모이드 함수 그래프 그리기
x = np.arange(-5.0, 5.0, 1.0)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y 축 범위 지정
plt.show()