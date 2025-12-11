import os
import json
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DigitDataset(data.Dataset):
    """Класс для формирования датасета MNIST"""
    def __init__(self, path, train=True, transform=None):
        # os.path.join() — это функция, которая используется для правильного
        # объединения компонентов пути к файлу или каталогу
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform

        # открываем файл 'format.json' с названиями папок и метками классов:
        # "class_0": 0 и т.д.
        with open(os.path.join(path, 'format.json'), 'r') as f:
            self.format = json.load(f)

        self.length = 0 # размер выборки (начальное значение 0)
        self.files = [] # список с информацией о файлах изображений
        self.targets = torch.eye(10) # единичная матрица 10х10 (one-hot)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            # os.listdir() возвращает список всех файлов и подкаталогов
            # в указанном каталоге
            list_files = os.listdir(path)
            self.length += len(list_files)
            # в список self.files добавляются кортежи в формате:
            # (путь к файлу изображения, класс изображения)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target),
                                              list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        # преобразование изображения, вытягивание в одну строку и нормализация
        if self.transform:
            # ravel возвращает представление исходного тензора 
            # (по возможности), flatten всегда возвращает копию
            img = self.transform(img).ravel().float() / 255

        return img, t

    def __len__(self):
        return self.length


class DigitNN(nn.Module):
    """Класс полносвязной нейросети для классификации изображений.
    На выходе 10 нейронов, т.к. у нас 10 классов изображений"""
    def __init__(self, input_size=28 * 28, hidden_size=32, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        # nn.functional.relu - это функциональный вызов API к функции relu;
        # nn.ReLU() создает nn.Module, который можно добавить, например,
        # в nn.Sequential()
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


to_tensor = tfs.ToImage() # для преобразования изображения в тензор

# здесь при определении датасета передается относительный путь к папке
d_train = DigitDataset('dataset', transform=to_tensor)
d_test = DigitDataset('dataset', train=False, transform=to_tensor)

# проверка правильности работы датасетов
# img, target = d_train[10]
# length = len(d_train)
# print(length) # 60000 для обучающей выборки MNIST
# img.show() # просмотр изображения

# определим итераторы для извлечения обучающих и тестовых данных
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

model = DigitNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss() # для многоклассовой классификации
epochs = 2 # количество эпох

# переводим модель в режим обучения
model.train()
for _ in range(epochs):
    # leave=True оставляет предыдущий progressbar на экране
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        # CrossEntropyLoss следует передавать выходные значения сети без
        # применения какой-либо нелинейной функции активации, сначала
        # передается predict
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# сохраним веса модели
# st = model.state_dict()
# torch.save(st, 'model_weights.tar')

# переводим модель в режим тестирования
model.eval()
with torch.no_grad():
    for x_test, y_test in test_data:
        # y_pred имеет размерность (batch_size, 10) и заполнен числами
        # чтобы получить предсказанную метку класса в виде числа, нужно
        # применить argmax
        y_pred = model(x_test).argmax(dim=1)
        # y_test - это одномерный тензор вида [0, 0, 1, ... 0] - one-hot
        # поэтому нужно получить из тензора метку класса в виде числа
        y = y_test.argmax(dim=1)
        # доля верных классификаций
        Q = (y_pred == y).float().mean().item()

print(f'Доля верных классификаций {Q}')