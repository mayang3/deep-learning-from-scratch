import numpy as np

# 다차원 배열
B = np.array([[1, 2], [3, 4], [5, 6]])

"""
[[1 2]
 [3 4]
 [5 6]]
"""
print(B)

# 차원의 수 출력
print(np.ndim(B))  # 2
# 3행 2열을 의미
print(B.shape)  # (3, 2)

# (2,3)
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

# (3,2)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

# 행렬 곱 (내적)
print(np.dot(A, B))

""" 행렬 곱을 할 수 없는 경우 """
C = np.array([[1, 2], [3, 4]])

# print(np.dot(A, C))  # Error! shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

""" 차원이 다른 배열의 경우 """
A = np.array(([1, 2], [3, 4], [5, 6]))
print(A.shape)  # (3, 2)

B = np.array([7, 8])
print(B.shape)  # (2,)

print(np.dot(A, B))  # [23 53 83]

""" 행렬의 곱으로 신경망 구하기 """
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)

print(Y)  # [ 5 11 17]
