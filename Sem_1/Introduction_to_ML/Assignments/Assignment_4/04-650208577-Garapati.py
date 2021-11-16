import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(0)

def generate_training_data():
    X1 = np.random.multivariate_normal([2, 1], np.diag([0.4, 0.04]), 100)
    X2 = np.random.multivariate_normal([1, 2], np.diag([0.4, 0.04]), 100)
    X = np.concatenate((X1, X2), axis=0)
    return X


def get_clusters(X, C):
    classes = []
    for x in X:
        min_dis = 100000
        min_c = -1
        for i in range(C.shape[0]):
            dis = np.linalg.norm(x - C[i], 2)
            if dis < min_dis:
                min_dis = dis
                min_c = i
        classes.append(min_c)
    return X[np.array(classes) == 0], X[np.array(classes) == 1]


def k_means_objective_function(C, clusters):
    error = 0
    for i in range(C.shape[0]):
        error += (np.sum(np.square(np.linalg.norm(clusters[i] - C[i], 2, axis=1))))
    return error



def Programming_P1():
    K = 2
    X = generate_training_data()

    C = X[np.random.randint(0, X.shape[0], K)]
    new_C = []

    epoch = 0

    while epoch != 5:
        new_C = []
        clusters = get_clusters(X, C)
        print(k_means_objective_function(C, clusters))
        for cluster in clusters:
            new_C.append(np.mean(cluster, axis=0))

        # print(C, new_C)

        plt.plot(clusters[0][:, 0], clusters[0][:, 1], 'bo')
        plt.plot(clusters[1][:, 0], clusters[1][:, 1], 'ro')
        plt.plot(C[0][0], C[0][1], 'yX')
        plt.plot(C[1][0], C[1][1], 'gX')
        plt.show()

        C = np.array(new_C)
        epoch += 1


def gaussian_distribution_function(x, mu, cov):
    N = 1/(np.power(2*np.pi, x.shape[0]/2) * np.power(np.linalg.det(cov), 1/2)) * np.exp(-1/2 * np.dot((x - mu).T, np.dot(np.linalg.inv(cov), (x - mu))))
    return N


def Programming_P2():
    P = np.array([1.5, 1.5]).reshape(2, 1)
    X = generate_training_data()
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    means = gm.means_
    weights = gm.weights_
    covariances = gm.covariances_
    print('Mean:', means, means.shape)

    print('Weights:', weights, weights.shape)
    print('Covariance: ', covariances, covariances.shape)

    X_predict = gm.predict(X)

    print(X_predict)

    plt.plot(X[X_predict == 0, 0], X[X_predict == 0, 1], 'bo')
    plt.plot(X[X_predict == 1, 0], X[X_predict == 1, 1], 'ro')
    plt.plot(means[0][0], means[0][1], 'yX')
    plt.plot(means[1][0], means[1][1], 'gX')
    plt.show()

    print('Predicted class using gmm calssifier: ', gm.predict([[1.5, 1.5]]))

    resp = []

    for i in range(2):
        resp.append(np.count_nonzero(X_predict == i)/len(X_predict) * gaussian_distribution_function(P, means[i].reshape(2, 1), covariances[i]))

    print(resp)

    resp = [resp[i]/sum(resp) for i in range(2)]

    print('responsibility values:', resp)
    print('Predicted class value using resp values: ', np.argmax(resp))

if __name__ == '__main__':
    # Programming_P1()

    Programming_P2()
