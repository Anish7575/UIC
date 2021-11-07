# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(55)

# E = -log(1 - x - y) - logx - logy
def energy_function(w):
    return float(-np.log(1 - w[0] - w[1]) - np.log(w[0]) - np.log(w[1]))

def gradient(w):
    return 1/(1 - w[0] - w[1]) + np.array([-1/(w[0]), -1/(w[1])])

def hessian(w):
    return 1/((1 - w[0] - w[1])**2) + np.array([[float(1/((w[0])**2)), 0.0], [0.0, float(1/((w[1])**2))]])

def gradient_descent(W, eta, epsilon):
    print('Gradient Descent with eta = {}, epsilon = {}: \n'.format(eta, epsilon))
    E = []

    while (True):
        E.append(energy_function(W[-1]))

        # Converging criteria: When Energy function change is less than 1e-6
        if (len(E) > 1 and abs(E[-1] - E[-2]) <= epsilon):
            break

        g = gradient(W[-1])

        w_new = W[-1] - (eta * g)

        # Taking care of the model when weights go beyond the domain
        while (w_new[0] <= 0 or w_new[1] <= 0 or (w_new[0] + w_new[1] >= 1)):
            eta /= 10.0
            print('eta value changed to {}\n'.format(eta))
            w_new = W[-1] - (eta * g)

        W.append(w_new)

    print('Epochs taken for convergence: ', len(W) - 1, '\n')
    print('Initial weights: ', W[0],'\n')
    print('Final weights: ', W[-1], '\n')
    print('Final energy function: ', E[-1], '\n')

    plt.title('Plot of Weights vs Epochs for gradient descent')
    ax = plt.axes(projection="3d")
    ax.set_xlabel('Weight w_0')
    ax.set_ylabel('Weight w_1')
    ax.set_zlabel('Epochs')
    ax.plot3D([float(x[0]) for x in W], [float(x[1]) for x in W], [i for i in range(0, len(W))], 'r', label='Weights vs Epochs')
    plt.legend(title='Legend')
    plt.show()

    plt.title('Plot of Epochs vs Energy Function for gradient descent')
    plt.xlabel('Epochs')
    plt.ylabel('Energy Function')
    plt.plot([x for x in range(0, len(W))], E, 'r', label='Epochs vs Energy Function')
    plt.legend(title='Legend')
    plt.show()

def newton_method(W, eta, epsilon):
    print('Newton\'s method with eta = {}, epsilon = {}: \n'.format(eta, epsilon))
    E = []

    while (True):
        E.append(energy_function(W[-1]))

        # Converging criteria: When Energy function change is less than 1e-6
        if (len(E) > 1 and abs(E[-1] - E[-2]) <= epsilon):
            break

        g = gradient(W[-1])
        H = hessian(W[-1])

        w_new = W[-1] - (eta * np.dot(np.linalg.inv(H), g))

        # Taking care of the case when weights go beyond the defined domain
        while (w_new[0] <= 0 or w_new[1] <= 0 or (w_new[0] + w_new[1] >= 1)):
            eta /= 10.0
            print('eta value changed to {}'.format(eta))
            w_new = W[-1] - (eta * np.dot(np.linalg.inv(H), g))

        W.append(w_new)

    print('Epochs taken for convergence: ', len(W) - 1, '\n')
    print('Initial weights: ', W[0], '\n')
    print('Final weights: ', W[-1], '\n')
    print('Final energy function: ', E[-1], '\n')

    plt.title('Plot of Weights vs Epochs for Newton\'s method')
    ax = plt.axes(projection="3d")
    ax.set_xlabel('Weight w_0')
    ax.set_ylabel('Weight w_1')
    ax.set_zlabel('Epochs')
    ax.plot3D([float(x[0]) for x in W], [float(x[1]) for x in W], [i for i in range(0, len(W))], 'r', label='Weights vs Epochs')
    plt.legend(title='Legend')
    plt.show()

    plt.title('Plot of Epochs vs Energy Function for Newton\'s method')
    plt.xlabel('Epochs')
    plt.ylabel('Energy Function')
    plt.plot([x for x in range(0, len(W))], E, 'r', label='Epochs vs Energy Function')
    plt.legend(title='Legend')
    plt.show()

if __name__ == '__main__':
    W1 = []
    W1.append(np.random.rand(2, 1)/2)
    eta = 1.0
    epsilon = 1e-6
    gradient_descent(W1, eta, epsilon)

    # Training with Newton's method with the same initial weights as gradient descent
    W2 = []
    W2.append(W1[0])
    newton_method(W2, eta, epsilon)
