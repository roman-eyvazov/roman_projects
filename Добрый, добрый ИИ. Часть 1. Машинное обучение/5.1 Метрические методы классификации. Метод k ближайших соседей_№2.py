import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
n_feature = 5 # количество признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7, -2, 4, 6]
V1 = [[D1 * r1 ** abs(i - j) for j in range(n_feature)] for i in 
	   range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature + 1)) * 0.5
V2 = [[D2 * r2 ** abs(i - j) for j in range(n_feature)] for i in 
	   range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature + 1)) * -0.5
V3 = [[D3 * r3 ** abs(i - j) for j in range(n_feature)] for i in 
	   range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 100, 120, 90
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
								random_state=123, test_size=0.3, shuffle=True)

k = 5 # кол-во ближайших соседей

# в данном случае евклидова метрика без корня по условию задачи
distances = np.array([[np.sum(((x - x_train) ** 2)) for x_train in X_train]
						for x in X_test])
# метки классов, упорядоченные по возрастанию расстояния
y_asc = y_train[np.argsort(distances, axis=1)]
distances = np.sort(distances, axis=1) # сортируем расстояния по возрастанию

predict = [] # пустой список под результаты работы классификатора
# применяем цикл, т.к. затруднительно использовать np.unique для двумерного 
# массива
for row in y_asc:
	value, count = np.unique(row[:k], return_counts=True)
	predict.append(value[np.argmax(count)])

Q = np.mean(predict != y_test) # доля неверных классификаций

print(Q)