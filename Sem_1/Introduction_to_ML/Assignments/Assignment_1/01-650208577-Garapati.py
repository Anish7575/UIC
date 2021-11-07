import numpy as np
import matplotlib.pyplot as plt

np.random.seed(55)

# 2.) Programming
# P1
A = np.array([[1, 2, 3], [0, 1, 3], [1, 0, 8]])
B = np.array([[7, 9, 2], [3, 4, 3], [6, 1, 3]])

print('P1:')
# Matrix Multiplication of A and B
print('Matrix multiplication of A and B:\n', np.dot(A, B), '\n')

# Element wise multiplication of A and B
print('Element wise multiplication of A and B:\n', np.multiply(A, B), '\n')

# P2

print('P2:')
# Inverse matrix of A
print('Inverse matrix of A:\n', np.linalg.inv(A), '\n')

# Transpose matrix of A
print('transpose matrix of A:\n', np.transpose(A), '\n')

# Matrix multplication of transpose(A) and A
print('Matrix multiplication of transpose(A) and A:\n', np.dot(np.transpose(A), A), '\n')

# p3
a = np.array([1, -1, 0])
b = np.array([0, 1, 1])

print('P3:')
# L1 norm of a, b, a-b
print('L1 norm of a:', np.linalg.norm(a, 1))
print('L1 norm of b:', np.linalg.norm(b, 1))
print('L1 norm of a - b:', np.linalg.norm(a - b, 1), '\n')

# L2 norm of a, b, a-b
print('L2 norm of a:', np.linalg.norm(a, 2))
print('L2 norm of b:', np.linalg.norm(b, 2))
print('L2 norm of a - b:', np.linalg.norm(a - b, 2), '\n')

# P4
X = np.random.uniform(0, 1, size=(20, 1))
Y = np.sin(2*np.pi*X) + np.random.normal(0, 0.3)

plt.title('P4. Plot of points (X, Y) generated')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, Y, 'ro', label='(X, Y=sin(2*pi*x)+n)')
plt.legend(title='Legend')
plt.show()

# P5
X1 = np.random.multivariate_normal([1, 1], np.eye(2, 2), size=(10,))

X2 = np.random.multivariate_normal([1, 2], np.eye(2, 2), size=(10,))

X3 = np.random.multivariate_normal([2, 1], np.eye(2, 2), size=(10,))

plt.title('P5. Plot of generated points belonging to 3 categories')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='Class 0')
plt.plot(X2[:, 0], X2[:, 1], 'bo', label='Class 1')
plt.plot(X3[:, 0], X3[:, 1], 'go', label='Class 2')
plt.legend(title='Legend')
plt.show()

# 3.) Optional Programming
C = np.array([[1, 2, 1], [2, 1, 0], [1, 0, 8]])

# O1
print('O1:')
l, V = np.linalg.eig(C)
print('eigenvalues of matrix C:', l, '\n')
print('eigenvectors of matrix C:\n', V, '\n')

# Reconstructing C from l and V
print(np.diag(l))
C1 = np.dot(np.dot(V, np.diag(l)), np.linalg.inv(V))
print('Reconstructed matrix using diag(eigenvalues) and eigenvectors:\n', C1, '\n')

# O2
print('O2:')
U, S, Vh = np.linalg.svd(C)
print('Matrices from singular eigen decomposition of C:')
print(U, '\n')
print(S, '\n')
print(Vh, '\n')

C2 = np.dot(np.dot(U, np.diag(S)), Vh)
print('Reconstructed matrix using U, S, Vh:\n', C2)

