import torch
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard的SummaryWriter
import warnings
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')  # 忽略警告信息

# 创建一个名为'logs_one'的TensorBoard日志写入器
writer = SummaryWriter('logs_one')

for i in range(100):
    writer.add_scalar('my test', 3 * i, i)
writer.close()


for i in range(100):
    image_path = f'train/{i:01d}.jpg'
    # print(image_path)
    img_pil = Image.open(image_path)
    img_arr = np.array(img_pil)
    writer.add_image('train', img_arr, i,
                     dataformats='HW')

writer.close()


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
torchvision.datasets.FashionMNIST('./data',download = True,train = True,transform = transform)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
torchvision.datasets.FashionMNIST('./data',download = True,train = True,transform = transform)
train_set = torchvision.datasets.FashionMNIST('./data',download = True,train = True,transform = transform)
test_set = torchvision.datasets.FashionMNIST('./data',download = True,train = True,transform = transform)
train_loader = DataLoader(train_set, batch_size=64,shuffle = True)
dataiter = iter(train_loader)
images, labels = next(dataiter)
import matplotlib.pyplot as plt
_= plt.imshow(torch.squeeze(images[0]),cmap = 'gray')

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, True)
writer.add_image('64 images', img_grid)
writer.close()



# 定义数据转换操作，将图像转换为张量，并进行标准化处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# 加载FashionMNIST数据集，如果不存在则下载
train_set = torchvision.datasets.FashionMNIST('data',
                                              download=True,
                                              train=True,
                                              transform=transform,
                                              )
# 输出第一张训练集中的图像的形状
print(train_set[0][0].shape)

import random
random.seed(42)  # 设置随机种子，保证随机结果的可重复性

# 创建一个名为'logs_one'的TensorBoard日志写入器
writer = SummaryWriter('logs_one')

selected_images = []

# 从每个类别中选择90张图像，并将它们添加到selected_images列表中
for class_id in range(10):
    # 获取所有标签为class_id的图像
    class_images = [img for img, label in train_set if label == class_id]
    # 从该类别的图像中随机选择90张，并将它们添加到selected_images列表中
    selected_class_images = random.sample(class_images, 90)
    selected_images.extend(selected_class_images)

# 类别标签
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 定义一个函数，用于在Matplotlib中显示图像
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)  # 若输入的是单通道图像，取通道维度的均值
    img = img / 2 + 0.5  # 反归一化操作，将像素值从[0,1]变换回[0,255]的范围
    npimg = img.numpy()  # 将张量转换为NumPy数组
    if one_channel:
        plt.imshow(npimg, cmap="Greys")  # 使用灰度色彩空间显示单通道图像
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 显示多通道图像

# 创建一个图像网格，用于显示多张图像
img_grid = torchvision.utils.make_grid(selected_images, nrow=30)

# 将图像网格转换为PIL图像，并保存到本地
pil_img = transforms.ToPILImage()(img_grid)
pil_img.save("selected_images_grid.png")

# 使用matplotlib_imshow函数显示图像网格
matplotlib_imshow(img_grid, True)

# 将图像网格添加到TensorBoard日志中，命名为'image_one'
writer.add_image('image_one', img_grid)

# 关闭TensorBoard日志写入器
writer.close()
