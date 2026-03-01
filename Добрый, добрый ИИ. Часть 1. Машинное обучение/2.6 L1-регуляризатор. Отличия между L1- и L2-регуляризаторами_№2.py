import numpy as np
from matplotlib import pyplot as plt

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


# модель
def model(w, X):
    return X @ w


# функция потерь
def loss(w, X, y):
    return (X @ w - y) ** 2


# производная функции потерь
def dL(w, X, y):
    return 2 / batch_size * X.T @ (model(w, X) - y)


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N - 1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x) # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для w
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления EMA
batch_size = 20 # размер мини-батча (величина K = 20)

X = coord_x.reshape(-1, 1) ** np.arange(N) # матрица признаков
y = coord_y # целевые значения

Qe = loss(w, X, y).mean() # начальное значение среднего эмпирического риска
np.random.seed(0)

for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1) # индекс случайного образа
    # берем выборку из batch_size значений - можно делать это через range
    batch = range(k, k + batch_size)
    Q = loss(w, X[batch], y[batch]).mean()
    Qe = lm * Q + (1 - lm) * Qe
    weights = np.array([0, *w[1:]]) # w0 не регуляризуется
    w = w - eta * (dL(w, X[batch], y[batch]) + lm_l1 * np.sign(weights))

Q = loss(w, X, y).mean() # итоговое значение среднего эмпирического риска

print(f'Вектор параметров w = {w}')
print(f'Последнее значение скользящего среднего Qe = {Qe}')
print(f'Cреднее эмпирического риска (показатель качества) Q = {Q}')

# построение графика
plt.plot(coord_x, coord_y, lw=2, label='func')
plt.plot(coord_x, model(w, X), lw=1, linestyle='-.', label='approx')

plt.legend(loc='best')
plt.title(f'Аппроксимация полиномом {N - 1} степени')
plt.grid()
plt.show()