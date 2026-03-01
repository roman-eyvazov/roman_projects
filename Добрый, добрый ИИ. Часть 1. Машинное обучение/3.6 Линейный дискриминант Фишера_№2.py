import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
# здесь принято по условию задачи, что корреляция между соседними
# величинами r, а через одну - r * r
V = [[D, D * r, D * r * r], [D * r, D, D * r], [D * r * r, D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

X_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)]) # классы 0 и 1

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационной матрицы
a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = 1 / (2 * N) * a @ a.T

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1 # вероятности появления классов
Py2, L2 = 1 - Py1, 1 # и величины штрафов неверной классификации

alpha1, alpha2 = np.linalg.inv(VV) @ mm1, np.linalg.inv(VV) @ mm2
beta1, beta2 = np.log(L1 * Py1) - 0.5 * mm1 @ np.linalg.inv(VV) @ mm1, \
                      np.log(L2 * Py2) - 0.5 * mm2 @ np.linalg.inv(VV) @ mm2

print(alpha1, alpha2)
print(beta1, beta2)