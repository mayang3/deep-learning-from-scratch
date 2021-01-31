import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 이미지를 넘파이 배열로 변환해주는 라이브러리
# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 변환
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
