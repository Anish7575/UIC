# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# Using transpose(w) = D.transpose(X).inv(X.transpose(X)) since transpose(X).X is a singular matrix
def linear_least_square(D, X):
    return np.transpose(np.dot(D, np.dot(np.transpose(X), np.linalg.inv(np.dot(X, np.transpose(X))))))

def energy_function(D, X, W):
    return np.linalg.norm(D - np.dot(X, W), 2)

def gradient_descent(D, X, eta, epsilon):
    W = np.random.randn(2, 1)
    E = []
    epoch = 0
    while(True):
        E.append(energy_function(D, X, W))

        # Convergence criteria: When the change in cost function is less than 1e-6
        if (len(E) > 1):
            if (abs(E[-1] - E[-2]) <= epsilon):
                break
        for x, d in zip(X, D):
            W = W + eta * np.dot(x.reshape(2, 1), (d - np.dot(np.transpose(W), x.reshape(2, 1))))
        epoch += 1
    print('Gradient descent converged in {} epochs\n'.format(epoch))
    print('Final Energy function: ', E[-1], '\n')
    return W

if __name__ == '__main__':
    X = np.array([list(range(1, 51))]).reshape(50, 1)
    X = np.concatenate((np.ones(shape=(50, 1)), X), axis=1)
    Y = np.array([x[1] + np.random.uniform(-1, 1) for x in X]).reshape(50, 1)

    # Linear least squares fit
    print('Linear Least Squares fit: \n')
    W_o = linear_least_square(np.transpose(Y), np.transpose(X))
    print('Weights obtained for linear least squares fit: ', W_o, '\n')
    print('Final energy function: ', energy_function(Y, X, W_o), '\n')

    plt.title('Plot of points (x,y) and linear least squares fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot([x[1] for x in X], Y, 'bx', label='Points(x, y)')
    x_plot = np.linspace(1, 50)
    y_plot = W_o[0] + W_o[1] * x_plot
    plt.plot(x_plot, y_plot, 'r', label='y = w0 + w1*x')
    plt.legend(title='Legend')
    plt.show()

    # Linear least squares fit using gradient descent
    eta = 0.0001
    epsilon = 1e-6
    print('Linear least Squares fit using Gradient descent with eta = {}, epsilon = {}:\n'.format(eta, epsilon))
    W_go = gradient_descent(Y, X, eta, epsilon)
    print('Weights obtained for linear least squares using gradient descent: ', W_go, '\n')

    plt.title('Plot of points (x,y) and linear least squares fit using gradient descent')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot([x[1] for x in X], Y, 'bx', label='Points(x, y)')
    x_plot = np.linspace(1, 50)
    y_plot = W_go[0] + W_go[1] * x_plot
    plt.plot(x_plot, y_plot, 'r', label='y = w0 + w1*x')
    plt.legend(title='Legend')
    plt.show()
