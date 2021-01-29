import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)


print(relu(0.3))  # 0.3
print(relu(-0.1))  # 0.0

# relu 함수 그래프 그리기
x = np.arange(-5.0, 5.0, 1.0)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 5)  # y 축 범위 지정
plt.show()
