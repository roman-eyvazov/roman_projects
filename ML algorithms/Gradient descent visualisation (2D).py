import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_regression

# возьмем сгенерированный датасет для задачи регрессии с 1 признаком
X, y = make_regression(n_samples=1000, n_features=1, n_targets=1,
                       random_state=42)
lr = np.full((2), 0.1) # lr для градиентного спуска
iters = 50 # количество итераций
batch_size = 10 # размер батча для SGD

# добавим 1 в качестве признака
X = np.hstack((np.array([1] * X.shape[0]).reshape(-1, 1), X))

# пустые списки под веса для построения графиков
w_plot, w_plot_sgd_batch, w_plot_sgd = [], [], []

def gradient_descent(X, y, mode='All samples', lr=lr, iters=iters, 
                     batch_size=batch_size):
    # начальные значения весов
    w = np.random.rand(2) * 50 
    w_batch = w
    w_sgd = w

    # градиентный спуск по всей выборке
    if mode == 'All samples':
        for _ in range(iters):
            grad_Q = 2 / X.shape[0] * (X.T @ (X @ w - y))
            w = w - lr * grad_Q
            w_plot.append(w)
        
        return w_plot

    # SGD по батчам
    elif mode == 'SGD_batch':
        for _ in range(iters):
            batch_indices = np.random.choice(X.shape[0], size=batch_size,
                                             replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            grad_Q_batch = 2 / batch_size * (X_batch.T @ (X_batch @
                           w_batch - y_batch))
            w_batch = w_batch - lr * grad_Q_batch
            w_plot_sgd_batch.append(w_batch)

        return w_plot_sgd_batch

    # SGD с одним образом
    elif mode == 'SGD':
        for _ in range(iters):
            index = np.random.randint(0, X.shape[0])
            x_sgd = X[index]
            y_sgd = y[index]
            grad_L = 2 * (x_sgd * (x_sgd @ w_sgd - y_sgd))
            w_sgd = w_sgd - lr * grad_L
            w_plot_sgd.append(w_sgd)
            
        return w_plot_sgd


# преобразуем списки с весами в массивы
w_plot = np.array(gradient_descent(X, y, mode='All samples'))
w_plot_sgd_batch = np.array(gradient_descent(X, y, mode='SGD_batch'))
w_plot_sgd = np.array(gradient_descent(X, y, mode='SGD'))

# построение графика
# значения весов w для графика
w_range = np.linspace(-25, 50, 200)
w0, w1 = np.meshgrid(w_range, w_range)

# рассчитаем значение эмпирического риска для графика проходом по всей выборке
Q = 0
for i in range(len(X)):
    Q += (X[i][0] * w0 + X[i][1] * w1 - y[i]) ** 2

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), dpi=100)

# градиентный спуск по всей выборке
c = ax[0].contour(w0, w1, Q, 10, cmap='viridis')
line, = ax[0].plot(w_plot[:, 0], w_plot[:, 1])
c.clabel()
ax[0].set_xlabel('w0')
ax[0].set_ylabel('w1')
title = ax[0].set_title(f'Gradient descent')
ax[0].grid()

# SGD по батчам
c_batch = ax[1].contour(w0, w1, Q, 10, cmap='viridis')
line_batch, = ax[1].plot(w_plot_sgd_batch[:, 0], w_plot_sgd_batch[:, 1])
c_batch.clabel()
ax[1].set_xlabel('w0')
ax[1].set_ylabel('w1')
title_batch = ax[1].set_title(f'SGD with batch')
ax[1].grid()

# SGD с одним образом
c_sgd = ax[2].contour(w0, w1, Q, 10, cmap='viridis')
line_sgd, = ax[2].plot(w_plot_sgd[:, 0], w_plot_sgd[:, 1])
c_sgd.clabel()
ax[2].set_xlabel('w0')
ax[2].set_ylabel('w1')
title_sgd = ax[2].set_title(f'SGD')
ax[2].grid()

def update(frame):
    line.set_xdata(w_plot[:frame][:, 0])
    line.set_ydata(w_plot[:frame][:, 1])
    title.set_text(f'Gradient descent, iter={frame}')

    line_batch.set_xdata(w_plot_sgd_batch[:frame][:, 0])
    line_batch.set_ydata(w_plot_sgd_batch[:frame][:, 1])
    title_batch.set_text(f'SGD with batch, iter={frame}')

    line_sgd.set_xdata(w_plot_sgd[:frame][:, 0])
    line_sgd.set_ydata(w_plot_sgd[:frame][:, 1])
    title_sgd.set_text(f'SGD, iter={frame}')

    return line, title, line_batch, title_batch, line_sgd, title_sgd,


ani = FuncAnimation(fig=fig, func=update, frames=iters, interval=50,
                    repeat=True)

plt.tight_layout()
plt.show()