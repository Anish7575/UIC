import math

def func(x1, x2, x3, x4):
    return ((math.exp(x1)) / (math.exp(x2) + math.exp(x3) + math.exp(x4)))

print(func(17, 17, -54, 54))
print(func(-54, 17, -54, 54))
print(func(54, 17, -54, 54))
