import numpy as np
import matplotlib.pylab as plt

# 자연로그 그래프 구하기
x = np.arange(0.0, 1.0, 0.01)
y = np.log(x + 1e-7)
plt.ylim(-5, 0)
plt.plot(x, y)
plt.show()


# np.log() 함수에 0을 입력하면 -INF가 되어 더 이상 계산을 진행할 수 없기 때문에,
# 아주 작은 값을 더해서 절대 0이 되지 않도록 한다.
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# index 2 가 정답
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))  # 0.510825457099338


y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))  # 2.302584092994546