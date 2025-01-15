# Visualizing-the-FashionMNIST-dataset
In this article, you will use TensorBoard to visualize the FashionMNIST dataset and learn how to present image data, scalar data, etc. in a more intuitive way to better understand the model performance.
# TensorBoard and FashionMNIST Visualization Guide

## Installation
To start, install TensorFlow and TensorBoard:
```bash
pip install tensorflow tensorboard
```

## TensorBoard Basics

**TensorBoard Official Documentation:** [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)

### Steps to Use TensorBoard:

1. **Import TensorBoard Library:**
   Ensure you have TensorFlow and TensorBoard installed, then import the library:
   ```python
   from torch.utils.tensorboard import SummaryWriter
   ```

2. **Create `SummaryWriter` Object:**
   ```python
   writer = SummaryWriter('logs')
   ```

3. **Log Training Metrics:**
   ```python
   for epoch in range(num_epochs):
       train_loss = ...  # Compute loss
       writer.add_scalar('Train/Loss', train_loss, epoch)
   ```

4. **Visualize Model Structure:**
   ```python
   writer.add_graph(model, input_data)
   ```

5. **Log Parameters and Gradients:**
   ```python
   for name, param in model.named_parameters():
       writer.add_histogram(name, param, epoch)
       writer.add_histogram(name + '_grad', param.grad, epoch)
   ```

6. **Log Embeddings:**
   ```python
   writer.add_embedding(embedding, metadata)
   ```

7. **Log Image and Media Data:**
   ```python
   writer.add_image('Image', image, epoch)
   ```

8. **Close the Writer:**
   ```python
   writer.close()
   ```

## FashionMNIST Dataset Overview
FashionMNIST is a dataset for image classification tasks, containing grayscale images of clothing and accessories.

### Key Details:
1. **Categories:**
   - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot.

2. **Image Size:**
   - 28x28 pixels, grayscale.

3. **Dataset Split:**
   - 60,000 training images and 10,000 testing images.

4. **Purpose:**
   - Alternative to MNIST for benchmarking and research.

## TensorBoard Visualization with FashionMNIST

### Import Libraries
```python
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
```

### Log Scalars
```python
warnings.filterwarnings('ignore')
writer = SummaryWriter('logs_one')

for i in range(100):
    writer.add_scalar('my test', 3 * i, i)

writer.close()
```

### Log Images
```python
for i in range(100):
    image_path = f'train/{i:01d}.jpg'
    img_pil = Image.open(image_path)
    img_arr = np.array(img_pil)
    writer.add_image('train', img_arr, i, dataformats='HW')

writer.close()
```

### Load and Preprocess FashionMNIST Dataset
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Visualize one image
images, labels = next(iter(train_loader))
plt.imshow(torch.squeeze(images[0]), cmap='gray')
```

### Display Random 8x8 Images
```python
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)) if not one_channel else npimg, cmap="Greys")

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('64 images', img_grid)
writer.close()
```

### Display Structured 30x30 Images

#### Select Images and Create Grid
```python
import random

selected_images = []
for class_id in range(10):
    class_images = [img for img, label in train_set if label == class_id]
    selected_class_images = random.sample(class_images, 90)
    selected_images.extend(selected_class_images)

img_grid = torchvision.utils.make_grid(selected_images, nrow=30)
```

#### Save and Display
```python
pil_img = transforms.ToPILImage()(img_grid)
pil_img.save("selected_images_grid.png")

matplotlib_imshow(img_grid, True)
writer.add_image('image_one', img_grid)
writer.close()


