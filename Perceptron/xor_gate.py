from Perceptron.nand_gate import NAND
from Perceptron.or_gate import OR
from Perceptron.and_gate import AND


# XOR 게이트 파이썬 구현버전
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


print(XOR(0, 0))  # 0
print(XOR(0, 1))  # 1
print(XOR(1, 0))  # 1
print(XOR(1, 1))  # 0
