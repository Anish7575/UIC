import numpy as np

x = np.array([[1, 2], [3, 4]])

print(np.dot(x.T, x) + np.eye(2, 2))
