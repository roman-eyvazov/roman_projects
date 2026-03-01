import numpy as np
import matplotlib.pyplot as plt

# тестовая выборка
x_test = [(9, 6), (2, 4), (-3, -1), (3, -2), (-3, 6), (7, -3), (6, 2)]

A = np.array([[2, 1], [7, 1]]) # матрица системы для расчета w
b_ = np.array([0, 7]) # столбец свободных членов

k, b = np.linalg.solve(A, b_) # коэффициенты k и b прямой

w2 = 5 # домножим на 5, чтобы избавиться от дробей
w1 = -k * w2
w0 = -b * w2
w = np.array([w0, w1, w2], dtype=int)
print(f'w = {w}')

# матрица признаков (дополняем первой единицей для w0)
X = np.array([[1, *x_i] for x_i in x_test])
predict = np.sign(X @ w).tolist() # преобразуем в обычный список
print(f'Классы: {predict}')

# построение графика
x_scatter = [x_i[0] for x_i in x_test]
y_scatter = [x_j[1] for x_j in x_test]
x_plot = np.arange(min(x_scatter), max(x_scatter) + 1)
plt.scatter(x_scatter, y_scatter, c='r')
plt.plot(x_plot, k * x_plot + b, linestyle='--')

plt.grid()
plt.show()