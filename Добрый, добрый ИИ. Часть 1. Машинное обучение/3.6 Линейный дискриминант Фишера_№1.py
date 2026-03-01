import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = [1, -2] # вектор математического ожидания 1-го класса
mean2 = [1, 3] # вектор математического ожидания 2-го класса
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
X_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * (-1), np.ones(N)]) # классы -1 и +1

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационной матрицы (здесь делим на 2N, т.к. считаем для 
# всей обучающей выборки)
a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = 1 / (2 * N) * a @ a.T

# параметры для гауссовского байесовского классификатора
Py = 0.5 # вероятности появления классов
L = 1 # величины штрафов неверной классификации

b = lambda x, m, v, L, P: np.log(L * P) - 0.5 * m @ np.linalg.inv(v) @ m \
                          + x @ np.linalg.inv(v) @ m 
predict = np.zeros(len(X_train))

for i in range(len(X_train)):
    predict[i] = np.argmax([b(X_train[i], mm1, VV, L, Py), 
                            b(X_train[i], mm2, VV, L, Py)], 0) * 2 - 1

Q = np.sum(predict != y_train)

print(Q)

# построение графика
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
            label='+1')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1],
            label='-1')
plt.scatter(X_train[predict != y_train][:, 0], 
            X_train[predict != y_train][:, 1], color='red', label='mistakes')

plt.grid()
plt.legend(loc='best')
plt.show()