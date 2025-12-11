import os
import random
import json
import pygame

# генерируем 1000 изображений для train и 100 для test
train_data = {'total': 10000, 'dir': "train"}
test_data = {'total': 1000, 'dir': "test"}
total_bk = 10 # количество фоновых изображений
dir_out = 'dataset' # папка для формирования датасета
file_format = 'format.json'
# возможные цвета для генерируемых точек
colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

# если не существует папки dir_out - создать ее
if not os.path.exists(dir_out):
    os.mkdir(dir_out)
    # если не существует папки train - создать ее
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    # если не существует папки test - создать ее
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

# загружаем изображение sun (64x64) из файла c возвращением объекта типа Surface
sun = pygame.image.load("images/sun64.png")
# список фоновых изображений
backs = [pygame.image.load(f"images/back_{n}.png")
         for n in range(1, total_bk + 1)]

for info in (train_data, test_data):
    # пустой словарь для сопоставления файла и координат солнца
    sun_coords = dict()

    for i in range(1, info['total'] + 1):
        file_out = f"sun_reg_{i}.png"
        # выбираем случайное фоновое изображение
        im = random.choice(backs).copy()

        # создаем случайное количество точек (от 20 до 100)
        for _ in range(random.randint(20, 100)):
            x0 = random.randint(0, 256)
            y0 = random.randint(0, 256)
            # добавляем на фоновое изображение im точки разных цветов со 
            # случайно выбранными координатами (x0, y0) и радиусом 1
            pygame.draw.circle(im, random.choice(colors),
                               (x0, y0), 1)

        # координаты (x, y) расположения солнца на фоновом изображении
        # ограничиваем их на 32 пиксела с каждой стороны, чтобы солнце
        # поместилось полностью
        x = random.randint(32, 256 - 32)
        y = random.randint(32, 256 - 32)
        # прописываем в словаре sun_coords для соответствующего файла
        # координаты солнца
        sun_coords[file_out] = (x, y)
        # накладываем на фоновое изображение im изображение солнца sun
        # с координатами размещения (x - 32, y - 32)
        im.blit(sun, (x - 32, y - 32))

        # сохраняем файл
        pygame.image.save(im, os.path.join(dir_out, info['dir'], file_out))

    # открываем файл file_format и записываем туда словарь sun_coords
    fp = open(os.path.join(dir_out, info['dir'], file_format), "w")
    json.dump(sun_coords, fp)
    fp.close()