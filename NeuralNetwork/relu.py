import numpy as np


def relu(x):
    return np.maximum(0, x)


print(relu(0.3))  # 0.3
print(relu(-0.1))  # 0.0
