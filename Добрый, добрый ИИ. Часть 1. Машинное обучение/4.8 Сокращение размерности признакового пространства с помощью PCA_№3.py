import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 3


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
K = 10 # cтепень полинома для аппроксимации функции
# обучающая выборка для поиска коэффициентов модели
X = coord_x.reshape(-1, 1) ** np.arange(K)
y = coord_y

X_train = X[::2] # обучающая выборка (входы)
y_train = y[::2] # обучающая выборка (целевые значения)

F = 1 / len(X_train) * X_train.T @ X_train # матрица Грама
L, W = np.linalg.eig(F) # определение собственных чисел и собственных векторов
# здесь сортировка по возрастанию собств. чисел (условие задачи)
WW = sorted(zip(L, W), key=lambda lx: lx[0], reverse=False)
WW = np.array([w[1] for w in WW]) # оставляем в массиве только собств. векторы

G = X @ WW.T # умножается на WW.T, чтобы скалярное произведение было корректным
G = G[:, :7] # оставляем в матрице G только первые 7 признаков по условию задачи

XX_train = G[::2] # матрица из образов с новыми признаками G
w = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ y_train

# восстановление функции через матрицу G
predict = G @ w

# посмотрим, как влияет количество оставляемых признаков на качество 
# аппроксимации
fig, ax = plt.subplots(2, 5, figsize=(12, 8))

for i in range(K):
	G = X @ WW.T 
	G = G[:, :i + 1] 
	XX_train = G[::2]
	w = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ y_train
	predict = G @ w

	# построение графиков
	ax = ax.flatten()
	ax[i].plot(coord_x, func(coord_x), label='func')
	ax[i].plot(coord_x, predict, linestyle='-.', label='approx')
	ax[i].set_title(f'Для {i + 1} признаков')
	ax[i].legend(loc='best')
	ax[i].grid()

plt.show()