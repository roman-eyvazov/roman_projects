import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SegmentDataset(data.Dataset):
    """Класс для формирования датасета с необходимыми для сегментации
    изображениями. В папке images находятся полноцветные изображения, в папке
    masks - цветные изображения масок. Пиксели фона соответствуют белому цвету,
    различные части обхекта - своим отдельным цветам"""
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # атрибуты для директории images
        path = os.path.join(self.path, 'images')
        # список с именами файлов изображений
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda x: os.path.join(path, x), list_files))

        # атрибуты для директории masks
        path = os.path.join(self.path, 'masks')
        # список с именами файлов масок
        list_files = os.listdir(path)
        self.masks = list(map(lambda x: os.path.join(path, x), list_files))

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        # конвертируем в RGB на случай, если изображение имеет 
        # другое представление
        img = Image.open(path_img).convert('RGB')
        # маску конвертируем в Grayscale (градации серого), т.к. нужно будет
        # только разделение на фон и пиксели объекта
        mask = Image.open(path_mask).convert('L')

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            # присваиваем 1 всем пикселям объекта (их цвет не белый)
            mask[mask < 250] = 1
            # присваиваем 0 всем пикселям фона (они белые)
            mask[mask >= 250] = 0

        return img, mask

    def __len__(self):
        return self.length


class UNetModel(nn.Module):
    """Модель U-Net для семантической сегментации изображений"""
    class _TwoConvLayers(nn.Module):
        """Вспомогательный вложенный класс, включающий в себя два подряд
        сверточных слоя с BatchNorm и ReLU"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                       3, stride=1, padding=1, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(out_channels),
                                       nn.Conv2d(out_channels, out_channels, 
                                       3, stride=1, padding=1, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(out_channels)
                                       )
        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        """Класс, включающий в себя два сверточных слоя (класс _TwoConvLayers)
        и слой MaxPooling"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        """Класс, включающий в себя слой ConvTranspose2d для восстановления
        сигнала и два сверточных слоя (класс _TwoConvLayers)"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2,
                                                stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)
            return u

    # конструктор класса UNetModel
    # параметр in_channels определяет число каналов входных изображений
    # параметр num_classes определяет число выходных каналов для маски
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1, stride=1)

    def forward(self, x):
        # обратить внимание, что _EncoderBlock возвращает y, x (y - выход
        # c MaxPool2d, x - выход с Conv2d)
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)


class SoftDiceLoss(nn.Module):
    """Класс, описывающий функцию потерь DiceLoss, которая применяется в случае
    несбалансированных классов"""
    def __init__(self, smooth=1):
        super().__init__()
        # smooth - число для добавления в числитель и знаменатель (чтобы 
        # исключить деление на ноль и граничные значения 0 и 1)
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits - выход сети U-Net, targets - целевая переменная
        num = targets.size(0)
        # применяем сигмоиду, чтобы получить результат в терминах вероятности
        # (т.к. на выходе U-Net нет функции активации); значение меньше 0.5
        # означает фон, а больше – объект
        probs = nn.functional.sigmoid(logits)
        # вытягиваем тензоры по батчам
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
 
        score = 2 * (intersection.sum(1) + self.smooth) \
                / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


# преобразования для загружаемых изображений и масок
tr_img = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
tr_mask = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32)])

# сформируем обучающую выборку и модель
d_train = SegmentDataset(r'dataset_seg', transform_img=tr_img, 
                         transform_mask=tr_mask)
train_data = data.DataLoader(d_train, batch_size=2, shuffle=True)
model = UNetModel()

# оптимизатор и функции потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_1 = nn.BCEWithLogitsLoss()
loss_2 = SoftDiceLoss()

epochs = 10 # количество эпох
model.train()

for i in range(epochs):
    # переменные для расчета скользящего среднего
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_1(predict, y_train) + loss_2(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = (1 - 1 / lm_count) * loss_mean + 1 / lm_count * loss.item()
        train_tqdm.set_description(f'''Epoch [{i + 1}/{epochs}],
                                   loss_mean={loss_mean:.3f}''')

# сохраним веса модели
st = model.state_dict()
torch.save(st, 'model_unet_seg.tar')

# загрузим веса модели
# weights = torch.load('model_unet_seg.tar', weights_only=False)
# model.load_state_dict(weights)

# протестируем на одном изображении (для тестирования нужна картинка, размер
# которой делится на 2 ** k (где k — количество даунсемплингов в два раза)
img = Image.open(r'car_test.jpg').convert('RGB')
img = tr_img(img).unsqueeze_(0)

model.eval()
with torch.no_grad():
    p = model(img).squeeze_(0)

# нужно поменять оси, чтобы получить (H, W, channels); также
# конвертируем в массив numpy
x = nn.functional.sigmoid(p.permute(1, 2, 0))
x = x.detach().numpy() * 255
x = np.clip(x, 0, 255).astype('uint8')

plt.imshow(x, cmap='gray')
plt.show()