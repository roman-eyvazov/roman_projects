import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(0)

data_x = np.array([(48, 118), (74, 96), (103, 82), (135, 76), (162, 79),
                   (184, 97), (206, 111), (231, 118), (251, 118), (275, 110),
                   (298, 86), (320, 68), (344, 62), (376, 61), (403, 75),
                   (424, 95), (440, 114), (254, 80), (219, 85), (288, 66),
                   (260, 92), (201, 76), (162, 66), (127, 135), (97, 143),
                   (83, 160), (82, 177), (88, 199), (105, 205), (135, 208),
                   (151, 198), (157, 169), (153, 152), (117, 158), (106, 168),
                   (106, 185), (123, 188), (125, 171), (139, 163), (139, 183),
                   (358, 127), (328, 132), (313, 146), (300, 169), (300, 181),
                   (308, 197), (326, 206), (339, 209), (370, 199), (380, 184),
                   (380, 147), (343, 154), (329, 169), (332, 184), (345, 185),
                   (363, 159), (361, 177), (344, 169), (311, 175), (351, 89),
                   (134, 96)])

M = np.mean(data_x, axis=0) # вычисление средних по каждой координате
D = np.var(data_x, axis=0) # вычисление дисперсий по каждой координате
K = 3 # число кластеров
# начальные центры кластеров
ma = [np.random.normal(M, np.sqrt(D / 10), 2) for n in range(K)]

metric = lambda x_i, x_k: np.sum(abs(x_i - x_k), axis=1) # манхэттенская метрика

# цветов должно быть не меньше кластеров (>= K)
colors = ('green', 'blue', 'black')
plt.ion() # интерактивный режим отображения графиков

for i in range(10): # выполняем 10 итераций
    # расстояния от каждой точки до каждого центроида
    distances = np.array([metric(data_x, ma[j]) for j in range(K)]).T
    idx_min = np.argmin(distances, axis=1) # номера кластеров для каждой точки
    cluster = [data_x[idx_min == v] for v in range(K)] # полученные кластеры

    ma = [np.mean(xx, axis=0) for xx in cluster] # пересчет центров кластеров

    plt.clf()
    # отображение найденных кластеров
    for i in range(K):
        xx = np.array(cluster[i]).T
        plt.scatter(xx[0], xx[1], s=10, color=colors[i])

    # отображение центров кластеров
    mx = [m[0] for m in ma]
    my = [m[1] for m in ma]
    plt.scatter(mx, my, s=50, color='red')

    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.05)

# ответ в виде списка - обязательное условие для проверки
X = [[tuple(sub) for sub in arr.tolist()] for arr in cluster]

plt.ioff()

# отображение найденных кластеров
for i in range(K):
    xx = np.array(cluster[i]).T
    plt.scatter(xx[0], xx[1], s=10, color=colors[i])

# отображение центров кластеров
mx = [m[0] for m in ma]
my = [m[1] for m in ma]
plt.scatter(mx, my, s=50, color='red')

plt.show()