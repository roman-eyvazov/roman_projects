import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = -0.5
D2 = 2.0
mean2 = [0, 2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 500
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * (-1), np.ones(N2)])

# вычисление оценок МО и ковариационных матриц
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = 1 / N1 * a @ a.T

a = (x2.T - mm2).T
VV2 = 1 / N2 * a @ a.T

# для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1 # вероятности появления классов
Py2, L2 = 1 - Py1, 1 # и величины штрафов неверной классификации

b = lambda x, m, v, L, P: np.log(L * P) - 0.5 * (x - m) @ np.linalg.inv(v) \
    @ (x - m).T - 0.5 * np.log(np.linalg.det(v))
predict = np.zeros(len(data_x))

for i in range(len(data_x)):
    predict[i] = np.argmax([b(data_x[i], mm1, VV1, L1, Py1),
                            b(data_x[i], mm2, VV2, L2, Py2)], 0) * 2 - 1

TP = np.sum((predict == +1) & (predict == data_y)) # true positive
TN = np.sum((predict == -1) & (predict == data_y)) # true negative
FP = np.sum((predict == +1) & (predict != data_y)) # false positive
FN = np.sum((predict == -1) & (predict != data_y)) # false negative

print(TP + TN + FP + FN) # проверка - должно быть равно (N1 + N2)

# построение графика
plt.scatter(data_x[data_y == 1][:, 0], data_x[data_y == 1][:, 1],
            label='+1')
plt.scatter(data_x[data_y == -1][:, 0], data_x[data_y == -1][:, 1],
            label='-1')
plt.scatter(data_x[predict != data_y][:, 0],
            data_x[predict != data_y][:, 1], label='mistakes', color='red')

plt.grid()
plt.legend(loc='best')
plt.show()