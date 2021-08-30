import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


# f: 최적화 하려는 함수
# init_x : 초깃값
# lr : learning rate 를 의미하는 학습률
# step_num : 경사법에 따른 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        # 함수의 기울기를 구한다.
        grad = numerical_gradient(f, x)
        # 기울기에 학습률을 곱한 값으로 매개변수 값을 step_num 번 갱신한다.
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()