# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

def unit_activation(W, X):
    return (np.dot(np.transpose(W), X) >= 0)


def count_misclassifications(S_0, S_1, W):
    misclass = 0
    for vector in S_0:
        misclass += (unit_activation(W, vector) == 1)
    for vector in S_1:
        misclass += (unit_activation(W, vector) == 0)
    return int(misclass)


def PTA(eta, W_train_init, S_0, S_1):
    W_train = W_train_init
    epoch = 0
    misclassifications = []
    while (1):
        misclassifications.append(count_misclassifications(S_0, S_1, W_train))

        if (misclassifications[-1] == 0):
            break
        for vector in S_1:
            W_train = W_train + eta * vector * \
                (1 - unit_activation(W_train, vector))
        for vector in S_0:
            W_train = W_train + eta * vector * \
                (0 - unit_activation(W_train, vector))
        epoch += 1

    return W_train, epoch, misclassifications


def assignment_2_experiment(n, W):
    # Generating a dataset of size n
    S = []
    for _ in range(0, n):
        X = np.random.uniform(-1, 1, size=(2, 1))
        X = np.insert(X, 0, 1, axis=0)
        S.append(X)

    # Classifying dataset into classes 1 and 0
    S_1 = []
    S_0 = []
    for vector in S:
        local_field = np.dot(np.transpose(W), vector)
        if (local_field >= 0):
            S_1.append(vector)
        else:
            S_0.append(vector)

    plt.title('Plots of Boundary and points (x1, x2) from classes 1 and 0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot([x[1] for x in S_1], [x[2] for x in S_1], 'x', label='Class 1')
    plt.plot([x[1] for x in S_0], [x[2] for x in S_0], 'o', label='Class 0')

    x_1 = np.linspace(-1, 1)
    x_2 = (-W[0] - W[1]*x_1)/W[2]
    plt.plot(x_1, x_2, 'r', label='Boundary')
    plt.legend(title='Legend', bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.show()

    print('----------------------------------------------------------------')

    # Generating initial training weights
    W_train_init = np.random.uniform(-1, 1, size=(3, 1))

    for eta in [1, 10, 0.1]:
        # Running the PTA algorithm with a specific eta
        W_train_comp, epoch, misclassifications = PTA(eta, W_train_init, S_0, S_1)

        print('Model training completed with eta({}), n({}) in {} epochs'.format(
            eta, n, epoch))
        print('Initial W_train:', W_train_init)
        print('Trained W_train:', W_train_comp)
        print('Optimal W:', W)

        plt.title(
            'Plot for Epoch vs Misclassifications with eta({}) and n({})'.format(eta, n))
        plt.xlabel('Epoch number')
        plt.ylabel('Misclassifications')
        plt.plot([i for i in range(0, epoch + 1)], misclassifications,
                 'r', label='epoch vs misclassifications')
        plt.legend(title='Legend')
        plt.show()
        print('----------------------------------------------------------------')

if (__name__ == '__main__'):
    # Choosing optimal weights
    w_0 = np.random.uniform(-0.25, 0.25)
    W = np.random.uniform(-1, 1, size=(2, 1))
    W = np.insert(W, 0, w_0, axis=0)

    assignment_2_experiment(n=100, W=W)
    assignment_2_experiment(n=1000, W=W)
