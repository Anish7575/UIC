import numpy as np  
import matplotlib.pyplot as plt  
  
def generate_data():
    X = np.random.uniform(0, 1, (100,1))
    y = 3 * X + 2 + np.random.randn(100, 1)*0.3

    return X, y

def learning(X, y):
    X_transpose = X.T  
    estimated_params = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
      
    return estimated_params

X, y = generate_data()  

plt.plot(X, y, "r.")
plt.axis([0, 1, 0, 6]) # x axis range 0 to 1, y axis range 0 to 6
plt.show()

X_b = np.concatenate((np.ones((100, 1)), X), axis=1)
theta = learning(X_b, y)
print(theta)

# draw the regressed line
  
test_X = np.array([[0], [1]])
test_X_b = np.concatenate((np.ones((2, 1)), test_X), axis=1)

prediction = test_X_b.dot(theta)

print(prediction)  

plt.plot(test_X, prediction, "r--")  
plt.plot(X, y, "b.")  
plt.axis([0, 1, 0, 6]) # x axis range 0 to 1, y axis range 0 to 6
plt.show()
