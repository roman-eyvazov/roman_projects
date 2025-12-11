import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs
from PIL import Image
import matplotlib.pyplot as plt

file_path = r'img.jpg'
img = Image.open(file_path)
# img.show()

# используем класс Compose, чтобы задать необходимые трансформации
# добавляем внешнюю ось batch_size
transforms = tfs.Compose([tfs.Grayscale(),
                          tfs.PILToTensor(),
                          tfs.ToDtype(torch.float32, scale=True),
                          tfs.Lambda(lambda x: x.unsqueeze_(dim=0))
                          ])

model = nn.Sequential(
                     nn.Conv2d(1, 16, kernel_size=3),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2),
                     nn.Conv2d(16, 32, kernel_size=3),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2),
                     nn.Conv2d(32, 64, kernel_size=3),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2)
                     )

t_in = transforms(img)

model.eval()
with torch.no_grad():
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8), dpi=100)
    ax = ax.flatten()
    # пройдемся по слоям сети в цикле
    for i in range(len(model)):
        layer = model[i]
        t_in = layer(t_in)
        
        # выход третьего слоя MaxPool2d
        if i == 8:
            for j, im in enumerate(torch.squeeze(t_in, 0)):
                # im - двумерный тензор, при подаче в ToPILImage 
                # двумерного тензора применяется цветовое кодирование LAB -
                # L (яркость), A (зелёный-пурпурный) и B (синий-жёлтый)
                # поэтому используем cmap='gray' при выводе
                img_out = tfs.ToPILImage()(im)
                ax[j].imshow(img_out, cmap='gray')
                ax[j].axis('off') # убираем элементы осей
             
plt.tight_layout()
plt.show()