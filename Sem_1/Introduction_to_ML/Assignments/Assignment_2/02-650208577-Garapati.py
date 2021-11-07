# Name: Sai Anish Garapati
# UIN: 650208577

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

def generate_data_for_order(X, M):
    train_X = np.concatenate((np.ones(shape=X.shape), X), axis=1)
    for i in range(2, M + 1):
        train_X = np.concatenate((train_X, np.power(X, i)), axis=1)
    return train_X

# l is 0 for linear regression and will be non-zero for ridge regression
def linear_regression(train_X, train_Y, l=0):
    return np.dot(np.linalg.inv(np.dot(train_X.T, train_X) + l*np.eye(train_X.T.shape[0], train_X.shape[1])), np.dot(train_X.T, train_Y))

def predict_Y(W, X):
    test_size = X.shape[0]
    test_X = np.concatenate((np.ones(shape=(test_size, 1)), X, np.power(X, 2), np.power(X, 3)), axis=1)
    return np.dot(test_X, W)

def root_mean_square_error(N, W, X, Y):
    return np.sqrt(1/N * np.linalg.norm(np.dot(X, W) - Y, 2))

def k_cross_fold_validation(train_X, Y, k, p, l=0):
    avg_prediction_error = 0
    W_learned_all_folds = []
    prediction_error_all_folds = []

    for i in range(0, k):
        W_learned_fold = linear_regression(np.delete(train_X, np.s_[p*i:p*i + p], axis=0), np.delete(Y, np.s_[p*i:p*i + p], axis=0), l)
        prediction_error_fold = root_mean_square_error(p, W_learned_fold, train_X[p*i:p*i + p], Y[p*i:p*i + p])
        W_learned_all_folds.append(W_learned_fold)
        prediction_error_all_folds.append(prediction_error_fold)
        avg_prediction_error += prediction_error_fold
    avg_prediction_error /= k
    return W_learned_all_folds, prediction_error_all_folds, avg_prediction_error

def ridge_regression(X, Y, l):
    return linear_regression(X, Y, l)


def Programming_P1(X, Y):
    train_X_poly3 = generate_data_for_order(X, 3)
    W_learned = linear_regression(train_X_poly3, Y)
    print('Learned parameters from Linear Regression: \n', W_learned, '\n')

    print('Learned polynomial function: {} + {}x + {}x^2 + {}x^3'.format(W_learned[0][0], W_learned[1][0], W_learned[2][0], W_learned[3][0]), '\n')

    test_X = np.array([[0], [0.25], [0.5], [0.75], [1]])
    predicted_Y = predict_Y(W_learned, test_X)
    for x, y in zip(test_X, predicted_Y):
        print('Prediction made on x={}: {}'.format(x[0], y[0]))
    print('\n')

    plt.title('Plots for Programming P1')
    plt.xlabel(('X'))
    plt.ylabel('Y')
    sine_plot_x = np.arange(0, 1, 0.01)
    plt.plot(sine_plot_x, np.sin(2*np.pi*sine_plot_x), label='noiseless sine plot')

    plt.plot(X, Y, 'yo', label='training data generated')

    plt.plot(test_X, predicted_Y, 'gx', label='(x_1, predicted_y(x_1))')

    plot_x = np.linspace(0, 1, num=50).reshape(50, 1)
    polynomial_x = generate_data_for_order(plot_x, 3)
    polynomial_equation = np.dot(polynomial_x, W_learned)

    plt.plot(plot_x, polynomial_equation, 'r', label='learned 3rd order polynomial')
    plt.legend(title='Legend')
    plt.show()


def Programming_P2(X, Y):
    k = 5
    p = int(20/k)
    train_X_poly3 = generate_data_for_order(X, 3)
    W_learned_all_folds, prediction_error_all_folds, avg_prediction_error = k_cross_fold_validation(train_X_poly3, Y, k, p)

    for W, error, fold in zip(W_learned_all_folds, prediction_error_all_folds, [i for i in range (1, 6)]):
        print('Learned Parameters on Linear regression for fold {}:'.format(fold))
        print(W)
        print('Prediction error on Linear regresssion for fold {}:'.format(fold), error, '\n')

    print('Average prediction error over all folds: ', avg_prediction_error)
    print('\n')


def Programming_P3(X, Y):
    k = 5
    p = int(20/k)
    W_learned_all_orders = []
    prediction_error_all_orders = []
    avg_prediction_error_all_orders = []

    for i in [1, 3, 5, 7, 9]:
        train_X_poly = generate_data_for_order(X, i)
        W_learned_fold, prediction_error, avg_prediction_error = k_cross_fold_validation(train_X_poly, Y, k, p)
        W_learned_all_orders.append(W_learned_fold)
        prediction_error_all_orders.append(prediction_error)
        avg_prediction_error_all_orders.append(avg_prediction_error)

    for avg_error, i in zip(avg_prediction_error_all_orders, [1, 3, 5, 7, 9]):
        print('Average cross-validation error for polynomial function of order {}: '.format(i), avg_error)

    print('Optimal order for polynomial function: ', 2*np.argmin(avg_prediction_error_all_orders) + 1)
    print('\n')

    plt.title('Plots for Programming P3: ')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y, 'o', label='Training data')
    plot_x = np.linspace(0, 1, num=50).reshape(50, 1)
    for i in range(0, 5):
        polynomial_x = generate_data_for_order(plot_x, 2*i + 1)
        polynomial_equation = np.dot(polynomial_x, W_learned_all_orders[i][0])
        plt.plot(plot_x, polynomial_equation, label='Curve for order M={}'.format(2*i + 1))
    plt.legend(title='Legend')
    plt.show()

    return W_learned_all_orders


def optional_O1(X, Y, W_learned_all_orders):
    l = 0.001
    M = 9
    train_X = generate_data_for_order(X, M)
    W_learned_ridge = ridge_regression(train_X, Y, l)

    print('\n')

    plt.title('Plots for Optional O1')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y, 'o', label='Training data')

    plot_x = np.linspace(0, 1, num=50).reshape(50, 1)
    polynomial_x = generate_data_for_order(plot_x, 9)
    polynomial_equation = np.dot(polynomial_x, W_learned_ridge)
    plt.plot(plot_x, polynomial_equation, label='Curve with ridge regression')

    polynomial_equation = np.dot(polynomial_x, W_learned_all_orders[4][0])
    plt.plot(plot_x, polynomial_equation, label='Curve with linear regression')
    plt.legend(title='Legend')
    plt.show()


def optional_O2(X, Y):
    # lambda values taken for testing
    L = [0.00001, 0.0001, 0.001, 0.05, 0.1, 1]
    M = 9
    k = 5
    p = int(20/k)
    train_X = generate_data_for_order(X, M)

    W_learned_all_lambda = []
    prediction_error_all_lambda = []
    avg_prediction_error_all_lamda = []

    for l in L:
        W_learned_all_folds, prediction_error_all_folds, avg_prediction_error = k_cross_fold_validation(train_X, Y, k, p, l)
        W_learned_all_lambda.append(W_learned_all_folds)
        prediction_error_all_lambda.append(prediction_error_all_folds)
        avg_prediction_error_all_lamda.append(avg_prediction_error)

    for avg_error, l in zip(avg_prediction_error_all_lamda, L):
        print('Average prediction error of lambda={}:'.format(l), avg_error)
    print('Optimal lambda over {}:'.format(L), L[np.argmin(avg_prediction_error_all_lamda)])

    plt.title('Plots for Optional O2')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y, 'o', label='Training data')

    plot_x = np.linspace(0, 1, num=50).reshape(50, 1)
    polynomial_x = generate_data_for_order(plot_x, M)
    for i in range(0, len(L)):
        polynomial_equation = np.dot(polynomial_x, W_learned_all_lambda[i][0])
        plt.plot(plot_x, polynomial_equation, label='With lambda = {}'.format(L[i]))
    plt.legend(title='Legend')
    plt.show()


if __name__ == '__main__':
    X = np.random.uniform(0, 1, size=(20, 1))
    Y = np.sin(2*np.pi*X) + np.random.normal(0, 0.3, size=(20, 1))

    print('Results for Programming Part 1:')
    Programming_P1(X, Y)

    print('Results for Programming Part 2:')
    Programming_P2(X, Y)

    print('Results for Programming Part 3:')
    W_learned_all_orders = Programming_P3(X, Y)

    print('Results for Optional Programming Part 1:')
    optional_O1(X, Y, W_learned_all_orders)

    print('Results for Optional Programming Part 2:')
    optional_O2(X, Y)
