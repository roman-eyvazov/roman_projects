import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
y_test = np.array([1, 1, 1, -1, -1, -1, -1])

w = np.array([-8/3, -2/3, 1])

# приведем к стандартному виду матрицы признаков
X_test = np.array([[1, x[0], x[1]] for x in x_test])
M = X_test @ w * y_test # margin (отступ)
Q = M[M < 0].size # расчет Q через скобки Айверсона

print(Q)