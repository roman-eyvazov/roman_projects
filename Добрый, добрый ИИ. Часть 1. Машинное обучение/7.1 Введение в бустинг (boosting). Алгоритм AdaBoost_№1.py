import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i - j) for j in range(n_feature)] for i in
       range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i - j) for j in range(n_feature)] for i in
       range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * (-1), np.ones(N2)])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                random_state=123, test_size=0.3, shuffle=True)

T = 10 # количество моделей
max_depth = 3 # максимальная глубина деревьев
# начальные значения весов для объектов выборки
w = np.ones(len(X_train)) / len(X_train)
models = [] # пустой список под полученные модели
alfa = [] # пустой список под веса для композиции

for i in range(T):
    # создаем и обучаем решающее дерево с весами объектов w
    models.append(DecisionTreeClassifier(criterion='gini', max_depth=max_depth))
    models[i].fit(X_train, y_train, sample_weight=w)

    # формируем прогнозы полученного дерева по обучающей выборке
    predicted = models[i].predict(X_train)
    # количество неверных классификаций с учетом весов
    N = np.sum(w * (predicted != y_train))
    # вес для текущего алгоритма
    alfa.append(0.5 * np.log((1 - N) / N) if N != 0 else 
                np.log((1 - 1e-8) / 1e-8))

    # пересчитываем и нормируем веса объектов выборки
    w = w * np.exp(-y_train * alfa[i] * predicted)
    w = w / np.sum(w)

# на основе полученных моделей делаем прогноз
predict = alfa[0] * models[0].predict(X_test)
for j in range(1, T):
    predict += alfa[j] * models[j].predict(X_test)

predict = np.sign(predict)
Q = np.sum(predict != y_test) # количество неверных классификаций

print(Q)

# отображаем полученные результаты классификации
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 10, data[:, 0].max() + 10
    y_min, y_max = data[:, 1].min() - 10, data[:, 1].max() + 10
    return np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

xx, yy = get_grid(X_test)
predicted = alfa[0] * models[0].predict(np.c_[xx.ravel(),
                                        yy.ravel()]).reshape(xx.shape)
for n in range(1, T):
    predicted += alfa[n] * models[n].predict(np.c_[xx.ravel(),
                                             yy.ravel()]).reshape(xx.shape)

plt.pcolormesh(xx, yy, predicted, cmap='spring', shading='auto')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=5000 * w, cmap='spring',
            edgecolors='black', linewidth=1.5)
plt.show()

"""
# построение графика
plt.scatter(X_test[:, 0][predict == -1], X_test[:, 1][predict == -1],
            color='red', label='Класс -1') # класс -1
plt.scatter(X_test[:, 0][predict == 1], X_test[:, 1][predict == 1],
            color='blue', label='Класс +1') # класс +1
plt.title('Применение AdaBoost')
plt.xlabel('Значение x1')
plt.ylabel('Значение x2')
plt.legend(loc='best')
plt.grid()
plt.show()
"""