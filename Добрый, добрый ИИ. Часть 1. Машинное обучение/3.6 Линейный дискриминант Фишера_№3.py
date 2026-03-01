import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# исходные параметры распределений трех классов
mean1 = np.array([1, -2]) # вектор математического ожидания 1-го класса
mean2 = np.array([-3, -1]) # вектор математического ожидания 2-го класса
mean3 = np.array([1, 2]) # вектор математического ожидания 3-го класса

r = 0.5 # коэффициент корреляции (всех классов)
D = 1.0 # дисперсия признаков вектора x (всех классов)
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

X_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2]) # классы 0, 1 и 2

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

# вычисление ковариационной матрицы (здесь делим на 3N, т.к. считаем для 
# всей обучающей выборки)
a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T, (x3.T - mm3).T])
VV = 1 / (3 * N) * a @ a.T
VV_inv = np.linalg.inv(VV)

# параметры для линейного дискриминанта Фишера
Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

model = lambda alpha, beta, x: alpha @ x + beta

alpha1 = VV_inv @ mm1
alpha2 = VV_inv @ mm2
alpha3 = VV_inv @ mm3

beta1 = np.log(L1 * Py1) - 0.5 * mm1.T @ VV_inv @ mm1
beta2 = np.log(L2 * Py2) - 0.5 * mm2.T @ VV_inv @ mm2
beta3 = np.log(L3 * Py3) - 0.5 * mm3.T @ VV_inv @ mm3

predict = np.array([np.argmax([model(alpha1, beta1, x), model(alpha2, beta2, x),
                    model(alpha3, beta3, x)]) for x in X_train])

Q = sum(predict != y_train)

print(Q)

# построение графика
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='1')
plt.scatter(X_train[y_train == 2][:, 0], X_train[y_train == 2][:, 1], label='2')
plt.scatter(X_train[predict != y_train][:, 0],
            X_train[predict != y_train][:, 1], color='red', label='mistakes')

plt.grid()
plt.legend(loc='best')
plt.show()