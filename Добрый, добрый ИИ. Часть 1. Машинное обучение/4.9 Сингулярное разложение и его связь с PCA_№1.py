import numpy as np

np.random.seed(0)

n_total = 1000 # число образов выборки
n_features = 200 # число признаков

table = np.zeros(shape=(n_total, n_features))

for _ in range(100): # заполняем обучающую выборку случайными числами
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

F = 1 / len(table) * table.T @ table # матрица Грама
L, W = np.linalg.eig(F) # определение собственных чисел и собственных векторов
# сортируем по убыванию значимости собственных векторов
WW = sorted(zip(L, W), key=lambda lx: lx[0], reverse=True)
WW = np.array([w[1] for w in WW]) # оставляем в массиве только собств. векторы

data_x = table @ WW.T # новый набор признаков в пространстве векторов WW
# нужно исключить признаки, для которых lm < 0.01
mask = (np.sort(L)[::-1] < 0.01).sum()
data_x = data_x[:, :n_features - mask]

print(data_x.shape)