import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(-10, 10, 1000)
y = x
z = lambda x, y: x ** 2 + y ** 2 + 30 * np.sin(x)
dz_x = lambda x: 2 * x + 30 * np.cos(x) # производная z по x
dz_y = lambda y: 2 * y # производная z по y

# для построения поверхности используем np.meshgrid()
x_grid, y_grid = np.meshgrid(x, y)
z_grid = z(x_grid, y_grid)

# начнем градиентный спуск со случайной точки
x0 = np.random.rand() * 10
y0 = x0
z0 = z(x0, y0)
lr = 0.05 # learning rate
iters = 50 # количество итераций

# пустой список под значения для обновления координат точки
array_plot = []

fig = plt.figure(figsize=(10, 6), dpi=100)
ax_3d = fig.add_subplot(projection='3d', elev=45)
ax_3d.plot_surface(x_grid, y_grid, z_grid, alpha=0.4)
point = ax_3d.scatter(x0, y0, z0, color='r')

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
title = ax_3d.set_title('Gradient descent')

for _ in range(iters):
    x0 = x0 - lr * dz_x(x0)
    y0 = y0 - lr * dz_y(y0)
    z0 = z(x0, y0)

    array_plot.append([x0, y0, z0])

def update(frame): # в данном случае frame - это range(iters)
    # почему-то не работает следующим образом, поэтому
    # приходится удалять и создавать оси ax_3d заново
    # point._offsets3d = (array_plot[frame][0], array_plot[frame][1],
    #                     array_plot[frame][2])
    # title.set_text(f'Gradient descent, iter={frame}')

    ax_3d.cla() # очистка осей ax_3d
    ax_3d.plot_surface(x_grid, y_grid, z_grid, alpha=0.4)
    ax_3d.scatter(array_plot[frame][0], array_plot[frame][1],
                  array_plot[frame][2], color='r')

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(f'Gradient descent, iter={frame}')
    
    #return point, title,


# если frames - целое число, это эквивалентно передаче range(frames) 
ani = FuncAnimation(fig=fig, func=update, frames=iters, interval=20,
                    repeat=False)

plt.show()