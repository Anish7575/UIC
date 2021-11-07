import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)


if __name__ == '__main__':
    n= 300
    N = 24
    X = np.random.uniform(0, 1, size=(n, 1))
    V = np.random.uniform(-0.1, 0.1, size=(n, 1))

    D = np.sin(20*X) + 3*X + V

    plt.plot(X, D, 'o')
    plt.show()
