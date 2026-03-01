import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7 # коэффициент корреляции признаков класса -1
D1 = 1.0 # дисперсия признаков класса -1
mean1 = [-1, -2, -1] # вектор математического ожидания класса -1
# здесь принято по условию задачи, что корреляция между соседними величинами
# r1, а через одну - r1 * r1
V1 = [[D1, D1 * r1, D1 * r1 * r1], [D1 * r1, D1, D1 * r1], 
      [D1 * r1 * r1, D1 * r1, D1]]

r2 = 0.5 # коэффициент корреляции класса +1
D2 = 2.0 # дисперсия признаков класса +1
mean2 = [1, 2, 1] # вектор математического ожидания класса +1
# здесь принято по условию задачи, что корреляция между соседними величинами
# r2, а через одну - r2 * r2
V2 = [[D2, D2 * r2, D2 * r2 * r2], [D2 * r2, D2, D2 * r2],
      [D2 * r2 * r2, D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
X_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * (-1), np.ones(N)])

# вычисление оценок математических ожиданий - mm1 (mm2) отличается от
# mean1 (mean2) несущественно - по сути, mean1 и mean2 - это "теоретические"
# МО, а mm1 и mm2 - "реальные" на основе сгенерированных выборок
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц - VV1 (VV2) отличается от V1 (V2)
# несущественно - по сути, V1 и V2 - это "теоретические" ковариационные
# матрицы, а VV1 и VV2 - "реальные" на основе сгенерированных выборок
a = (x1.T - mm1).T
VV1 = 1 / N * a @ a.T

a = (x2.T - mm2).T
VV2 = 1 / N * a @ a.T

# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1
Py2, L2 = 1 - Py1, 1

b = lambda x, m, v, L, P: np.log(L * P) - 0.5 * (x - m) @ np.linalg.inv(v) \
                          @ (x - m).T - 0.5 * np.log(np.linalg.det(v))
predict = np.zeros(len(X_train))

for i in range(len(X_train)):
    predict[i] = np.argmax([b(X_train[i], mm1, VV1, L1, Py1), 
                            b(X_train[i], mm2, VV2, L2, Py2)], 0) * 2 - 1

Q = np.sum(predict != y_train)

print(Q)

# построение графика
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
            label='+1')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1],
            label='-1')
plt.scatter(X_train[predict != y_train][:, 0],
            X_train[predict != y_train][:, 1], label='mistakes', color='red')

plt.grid()
plt.legend(loc='best')
plt.show()