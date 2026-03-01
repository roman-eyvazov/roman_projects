import numpy as np
from matplotlib import pyplot as plt

def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2 * x) + 1 * np.cos(4 * x) + 10


x = np.arange(-3.0, 4.1, 0.1) # значения по оси абсцисс с шагом 0.1
y = np.array(func(x)) # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N - 1)
lm = 20  # параметр лямбда для L2-регуляризатора

X = x.reshape(-1, 1) ** np.arange(N) # матрица входных векторов
#X  = np.array([[a ** n for n in range(N)] for a in x]) # либо так
IL = lm * np.eye(N) # матрица lambda * I
IL[0][0] = 0  # первый коэффициент не регуляризуем

X_train = X[::2] # обучающая выборка (входы)
y_train = y[::2] # обучающая выборка (целевые значения)

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ y_train
Q = np.mean((X @ w - y) ** 2)

# построение графика
plt.plot(x, y, lw=2, label='func')
plt.plot(x, X @ w, lw=1, linestyle='-.', label='approx')

plt.legend(loc='best')
plt.title(f'Аппроксимация полиномом {N - 1} степени')
plt.grid()
plt.show()