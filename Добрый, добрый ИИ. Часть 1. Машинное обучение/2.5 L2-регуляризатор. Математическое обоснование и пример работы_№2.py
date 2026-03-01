import numpy as np
from matplotlib import pyplot as plt

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# значения по оси абсцисс [-4; 6] с шагом 0.1
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x) # значения функции по оси ординат

N = 5 # сложность модели (полином степени N - 1)
lm_l2 = 2 # коэффициент λ для L2-регуляризатора
sz = len(coord_x) # количество точек
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для w
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение λ вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

X = coord_x.reshape(-1, 1) ** np.arange(N) # матрица признаков
y = coord_y # целевые значения

np.random.seed(0)
Qe = np.mean((X @ w - y) ** 2) # начальное значение среднего эмпирического риска

# градиентный спуск с регуляризацией
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1) # индекс случайного образа
    index = k + batch_size # берем выборку из batch_size значений
    Q = np.mean((X[k:index] @ w - y[k:index]) ** 2) # эмпирический риск
    Qe = lm * Q + (1 - lm) * Qe
    weights = np.array([0, *w[1:]]) # w0 не регуляризуется
    grad_Q = 2 / batch_size * (X[k:index].T @ (X[k:index] @ w - y[k:index]))
    w = w - eta * (grad_Q + lm_l2 * weights) # обновление с учётом регуляризации

Q = np.mean((X @ w - y) ** 2) # итоговое значение среднего эмпирического риска

print(f'Вектор параметров w = {w}')
print(f'Последнее значение скользящего среднего Qe = {Qe}')
print(f'Cреднее эмпирического риска (показатель качества) Q = {Q}')

# построение графика
plt.plot(coord_x, coord_y, label='func')
plt.plot(coord_x, X @ w, linestyle='-.', label='approx')

plt.legend(loc='best')
plt.title(f'Аппроксимация полиномом {N - 1} степени')
plt.grid()
plt.show()