# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(2021)


def generate_random_points(size, low, high):
    data = (high - low) * np.random.random_sample((size, 2)) + low
    return data


def generate_training_data(N, l1, h1, l2, h2):
    X1 = generate_random_points(N, l1, h1)
    Y1 = np.ones(N)
    X2 = generate_random_points(N, l2, h2)
    Y2 = np.zeros(N)
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)
    indices = np.arange(2*N)
    np.random.shuffle(indices)
    return X[indices, :], Y[indices]


def KNN_classifier(K, X, Y, test_sample):
    distances = np.sum((X - test_sample)**2, axis=1)
    arg_distances = np.argsort(distances)
    prediction = np.argmax(np.bincount([Y[arg] for arg in arg_distances[:K]]))

    return prediction, arg_distances


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def loss(output, Y):
    return (-Y * np.log(output) - (1 - Y) * np.log(1 - output)).mean()


def logistic_regression(X, Y, lr=0.01, epochs=100000):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    W = np.zeros((X.shape[1], 1))
    for i in range(epochs):
        output = sigmoid(np.dot(X, W))
        gradient = np.dot(X.T,  (output - Y)) / Y.shape[0]
        W -= lr * gradient

        if i % 10000 == 0:
            output = sigmoid(np.dot(X, W))
            print('Loss at epoch {}: {}'.format(i, loss(output, Y)))
    return W

def predict(X, W):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    P = sigmoid(np.dot(X, W))
    return int(P >= 0.5)


def Programming_P1():
    N = 20
    X, Y = generate_training_data(N, l1=0, h1=1, l2=1, h2=2)
    K = 3
    test_sample = np.array([1.0, 1.0])
    prediction, arg_distances = KNN_classifier(K, X, Y, test_sample)

    print('Predicted Class for {}: {}'.format(test_sample, prediction))
    print('K=3 nearest neighbors and corresponding classes:')

    for p, c in zip(X[arg_distances[:K]], Y[arg_distances[:K]]):
        print(p, c)

    plt.title('Plot of training data and test sample with nearest neighbors')
    plt.xlabel('x1')
    plt.ylabel('x2')
    X1 = np.array([X[i] for i in range(0, len(X)) if Y[i] == 1])
    X2 = np.array([X[i] for i in range(0, len(X)) if Y[i] == 0])
    plt.plot(X1[:, 0], X1[:, 1], 'rx', label='Class with Y = 1')
    plt.plot(X2[:, 0], X2[:, 1], 'bo', label='Class with Y = 0')
    plt.plot(test_sample[0], test_sample[1], 'gs', label='Test Sample')

    plt.plot(X[arg_distances[:K]][:, 0], X[arg_distances[:K]][:, 1], 'yX', label='K=3 nearest neighbors')
    plt.legend(title='Legend')
    plt.show()


def Programming_P2():
    N = 20
    X, Y = generate_training_data(N, l1=0, h1=1.5, l2=0.5, h2=2)
    K = [1, 3, 5, 7, 9]
    optim_k = [0, 0]
    folds = 5
    d = (2*N)/5
    for k in K:
        total_accuracy = 0
        for fold in range(0, folds):
            l, r = int(fold*d), int(fold*d + d)
            test_set, test_set_labels = X[l:r], Y[l:r]
            correct_predictions = 0
            for test_sample, test_label in zip(test_set, test_set_labels):
                pred = KNN_classifier(k, np.delete(X, np.s_[l:r], axis=0), np.delete(Y, np.s_[l:r], axis=0), test_sample)[0]
                correct_predictions += (pred == test_label)
            print('K: {}, Fold: {}, accuracy:'.format(k, fold + 1), correct_predictions/d)
            total_accuracy += (correct_predictions/d)
        print('K: {}, Avg_accuracy: {}\n'.format(k, total_accuracy/folds))
        if total_accuracy/folds >= optim_k[1]:
            optim_k = [k, total_accuracy/folds]
    print('Optimal K: {}\n'.format(optim_k[0]))


def Programming_P3():
    N = 20
    X, Y = generate_training_data(N, l1=0, h1=1, l2=1, h2=2)
    test_set = [[0.5, 0.5], [1, 1], [1.5, 1.5]]

    W_LR = logistic_regression(X, Y.reshape(len(Y), 1))
    print()
    print('Predictions using Logistic regression model:')
    for test_sample in test_set:
        print('Predicted class for {}: {}'.format(test_sample, predict(np.array(test_sample).reshape((1, 2)), W_LR)))
    print()

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, Y)
    W_SVM = clf.coef_
    b_SVM = clf.intercept_
    print('Predictions using SVM model:')
    for test_sample in test_set:
        print('Predicted class for {}: {}'.format(test_sample, int(clf.predict([test_sample]))))
    print()

    plt.title('Plot of training data with decision boundaries from logistic and SVM')
    plt.xlabel('x1')
    plt.ylabel('x2')
    X1 = np.array([X[i] for i in range(0, len(X)) if Y[i] == 1])
    X2 = np.array([X[i] for i in range(0, len(X)) if Y[i] == 0])
    plt.plot(X1[:, 0], X1[:, 1], 'rx', label='Class with Y = 1')
    plt.plot(X2[:, 0], X2[:, 1], 'bo', label='Class with Y = 0')

    X_plot = np.linspace(0, 2)
    plt.plot(X_plot, (-W_LR[0] - W_LR[1] * X_plot)/W_LR[2], 'g', label='Boundary with Logistic Regression')
    plt.plot(X_plot, (-b_SVM - W_SVM[0][0] * X_plot)/W_SVM[0][1], 'b', label='Boundary with SVM')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Programming P1\n')
    Programming_P1()
    print('----------------------------------------\n')

    print('Programming P2\n')
    Programming_P2()
    print('----------------------------------------\n')

    print('Programming P3\n')
    Programming_P3()
