import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


"""
예제
정답이 2 이기 때문에 index 2 에 가장 확률이 높다고 추정한 결과의 손실 함수쪽 출력이 적다 (즉 오차가 적다)

반대로 index 7 의 확률이 가장 높다고 추정한 
"""

# 정답은 '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1 : '2' 일 확률이 가장 높다고 추정함 (0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))  # 0.09750000000000003

# 예2 : '7'일 확률이 가장 높다고 추정함 (0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))  # 0.5975