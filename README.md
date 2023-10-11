# PyTorch Handwritten Digit Recognition

This project demonstrates handwritten digit recognition using PyTorch. It includes setting up the dataset, creating a convolutional neural network (CNN) model, optimizing it, and training the model. The code also evaluates the model's performance on a test dataset.

## Setting up and Importing the Dataset

We start by importing the necessary libraries and loading the MNIST dataset.

```python
from torchvision import datasets
from torchvision.transforms import ToTensor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)
```

## Analyzing and Exploring the Data

Let's analyze and explore the training and testing datasets:

### Training Data

- Dataset: MNIST
- Number of datapoints: 60,000
- Root location: data
- Split: Train
- Transform: ToTensor()

### Testing Data

- Dataset: MNIST
- Number of datapoints: 10,000
- Root location: data
- Split: Test
- Transform: ToTensor()

We also check the shape and size of the dataset:

```python
train_data.data.shape  # torch.Size([60000, 28, 28])
test_data.data.shape   # torch.Size([10000, 28, 28])
train_data.targets.shape  # torch.Size([60000])
```

## Creating Data Loader

We create data loaders for both training and testing data:

```python
from torch.utils.data import DataLoader

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
}
```

## Creating the Machine Learning Model

We define a simple Convolutional Neural Network (CNN) model for handwritten digit recognition:

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)
```

## Optimizing the Machine Learning Model using CUDA

We check if CUDA is available and move the model to the GPU if it is:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
```

## Creating the Dataset Training Mode

We define the training loop for the dataset:

```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders['train'].dataset)} ({100 * batch_idx / len(loaders['train']):0f}%)]\t{loss.item():.6f}")
```

## Creating the Dataset Testing Mode

We define the testing loop for the dataset:

```python
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f"\nTest set: Average loss: {test_loss: 0.4f}, Accuracy {correct}/{len(loaders['test'].dataset)}  ({100 * correct / len(loaders['test'].dataset):.0f}%\n")
```



## Training and Testing the Model

We train and test the model for a specified number of epochs:

```python
for epoch in range(1, 10):
    train(epoch)
    test()
```

## Conclusion

This is a simple example of how to perform handwritten digit recognition using PyTorch. You can further optimize the model and hyperparameters for better performance, and you can also explore more advanced deep learning models for this task.
