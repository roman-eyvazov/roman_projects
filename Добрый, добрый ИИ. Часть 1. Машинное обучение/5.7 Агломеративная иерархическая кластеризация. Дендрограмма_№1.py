import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

X = np.array([(189, 185), (172, 205), (156, 221), (154, 245), (164, 265),
			  (183, 275), (204, 276), (227, 271), (241, 255), (250, 229),
			  (240, 197), (217, 183), (194, 202), (179, 224), (179, 248),
			  (199, 249), (197, 227), (211, 214), (211, 242), (210, 265),
			  (226, 237), (218, 196), (79, 106), (97, 132), (117, 159),
			  (138, 174), (148, 163), (140, 145), (121, 123), (112, 108),
			  (89, 92), (282, 162), (298, 180), (344, 154), (344, 113),
			  (362, 67), (397, 77), (412, 121), (379, 112), (377, 148),
			  (312, 130)])

K = 3 # максимальное число выделяемых кластеров
# с использованием расстояния Уорда
clustering = AgglomerativeClustering(n_clusters=K, linkage="ward", 
									 metric="euclidean")
res = clustering.fit_predict(X)

# выделим образы разных кластеров для построения графика
X1 = X[res == 0]
X2 = X[res == 1]
X3 = X[res == 2]

print(clustering.n_clusters) # найденное число кластеров

# построение графика
plt.scatter(X1[:, 0], X1[:, 1], color='red')
plt.scatter(X2[:, 0], X2[:, 1], color='blue')
plt.scatter(X3[:, 0], X3[:, 1], color='green')
plt.title('Применение AgglomerativeClustering')
plt.xlabel('Значение x1')
plt.ylabel('Значение x2')

plt.grid()
plt.show()