# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

np.random.seed(2021)

def unit_activation(vector, W_train):
    local_field = np.dot(W_train, vector.reshape(784, 1))
    activated_output = np.array([int(lf >= 0) for lf in local_field]).reshape(10, 1)
    return activated_output

def calculate_errors(X_train, Y_train, W_train):
    errors = 0
    for vector, label in zip(X_train, Y_train):
        local_field = np.dot(W_train, vector.reshape(784, 1))
        if (local_field.argmax() != label):
            errors += 1
    return errors

def multicategory_PTA(n, eta, epsilon):
    X_train = train_set[:n, :]
    Y_train = train_labels[:n]
    W_train = np.random.randn(10, 784)
    epoch = 0
    errors = []
    print('Training model with n={}, eta={}, epsilon={}'.format(n, eta, epsilon))
    while (True):
        # Calculating misclassifications
        errors.append(calculate_errors(X_train, Y_train, W_train))

        # Condition to break for epsilon=0 for n=60000
        # if ((errors[-1]/n <= epsilon) or (epoch == 100)):
        #    break
        if (errors[-1]/n <= epsilon):
            break

        for vector, label in zip(X_train, Y_train):
            label_vector = np.zeros(shape=(10, 1))
            label_vector[label] = 1
            W_train = W_train + eta*np.dot((label_vector - unit_activation(vector, W_train)), vector.reshape(1, 784))
        epoch += 1
    print('Training model completed in {} epochs with {} errors(Error%: {}%) in the training set'.format(epoch, errors[-1], errors[-1]/n * 100))

    plt.title('Plot for Epoch vs Misclassifications with n({}), eta({}) and epsilon({})'.format(n, eta, epsilon))
    plt.xlabel('Epoch number')
    plt.ylabel('Misclassifications')
    plt.plot([i for i in range(0, epoch + 1)], errors, 'r', label = 'epoch vs misclassifications')
    plt.legend(title='Legend')
    plt.show()

    # Calculating Testset accuracy
    test_errors = calculate_errors(test_set, test_labels, W_train)
    print('Test Error percentage for n = {}, eta = {}, epsilon = {}: {}%'.format(n, eta, epsilon, test_errors/test_set.shape[0] * 100))
    print("\n")

if (__name__ == '__main__'):
    train_set, train_labels = loadlocal_mnist(
            images_path='train-images.idx3-ubyte',
            labels_path='train-labels.idx1-ubyte')
    test_set, test_labels = loadlocal_mnist(
            images_path='t10k-images.idx3-ubyte',
            labels_path='t10k-labels.idx1-ubyte')

    print('\n')
    multicategory_PTA(n=50, eta=1, epsilon=0)
    multicategory_PTA(n=1000, eta=1, epsilon=0)
    # multicategory_PTA(n=60000, eta=1, epsilon=0)
    # epsilon=0.13 is chosen depending on the result from above

    multicategory_PTA(n=60000, eta=1, epsilon=0.13)
    multicategory_PTA(n=60000, eta=10, epsilon=0.13)
    multicategory_PTA(n=60000, eta=0.1, epsilon=0.13)

