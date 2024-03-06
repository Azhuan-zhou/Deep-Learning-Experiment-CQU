from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from torchsummary import summary

plt.switch_backend('TkAgg')


# ----------------搭建数据集-----------------
class DataSet(Dataset):
    # 通过下标索引可以访问训练集的所有数据
    def __getitem__(self, index):
        dataset = self.training_set + self.training_set
        img, label = dataset[index]
        return img, label

    def __init__(self, root, batch_size):
        transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()]
        )
        # 加载数据
        self.training_set = MNIST(root,
                                  train=True,
                                  transform=transform,
                                  download=False
                                  )
        self.testing_set = MNIST(root,
                                 train=False,
                                 transform=transform,
                                 download=False
                                 )
        self.training_set_loader = DataLoader(self.training_set,
                                              batch_size=batch_size,
                                              shuffle=True)
        self.testing_set_loader = DataLoader(self.testing_set,
                                             batch_size=batch_size,
                                             shuffle=True)


# ---------显示数据集的图片--------------
def show(images, numRow, numColumn, title=None, scale=4):
    # 显示窗口尺寸
    figSize = (numRow * scale, numColumn * scale)
    _, axes = plt.subplots(numRow, numColumn, figsize=figSize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图像为张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        if torch.is_tensor(title):
            ax.set_title(title[i].item())
    return axes


class CovNet(nn.Module):
    def __init__(self, input_chanel, output_size):
        super(CovNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(input_chanel, 6, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(6, 16, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(16, 120, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(120, 84),
                                   nn.Dropout(p=0.3),
                                   nn.Linear(84, output_size),
                                   )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, trainIter, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    total = 0
    correct = 0
    for batch in trainIter:
        images = batch[0]
        labels = batch[1]
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad:
            # 模型预测的类别
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
    return epoch_loss / len(trainIter), 100 * correct / total


def draw(loss, epoch):
    x = range(epoch)
    plt.plot(x, loss, 'b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


# ---------------- testing ----------------------
def evaluate(model, Iter, criterion):
    model.eval()
    epoch_loss = 0
    # 统计分类对的个数
    correct = 0
    # 统计总共有多少样本
    total = 0
    with torch.no_grad():
        for batch in Iter:
            images = batch[0]
            labels = batch[1]
            outputs = model(images)
            # 模型预测的类别
            _, predicted = torch.max(outputs.data, 1)
            # 计算损失
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
    return epoch_loss / len(Iter), 100 * correct / total


def train(model, trainIter, validIter, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, trainIter, optimizer, criterion)
        validation_loss, validation_accuracy = evaluate(model, validIter, criterion)
        print('Epoch:[{}/{}]'.format(epoch + 1, epochs))
        print('Train Loss : {:.3f}  | Train Accuracy : {:.3f}%'.format(train_loss, train_accuracy))
        print('Validation Loss : {:.3f}  | Validation Accuracy : {:.3f}%'.format(validation_loss,
                                                                                 validation_accuracy))


if __name__ == '__main__':
    # -----------参数初始化-----------
    inputChanel = 1
    outputSize = 10
    epochs = 30
    learningRate = 0.01
    batch_size = 256
    device = torch.device('cuda')

    print('1.Loading data............')
    dataset = DataSet('../data', batch_size)
    # 训练数据
    trainLoader = dataset.training_set_loader
    # 测试数据
    testLoader = dataset.testing_set_loader

    # 显示数据集的部分图片
    # X, y = next(iter(dataset.training_set_loader))
    # axes = show(X.reshape(18, 32, 32), 2, 9, title=y)
    # plt.show()

    # 加载模型
    covNet = CovNet(inputChanel, outputSize).to(device)
    summary(covNet, (1, 32, 32))

    print('2.Training............')
    lossInEpoch = []
    train(covNet, trainLoader, epochs, learningRate, lossInEpoch)
    draw(lossInEpoch, epochs)

    print('3.testing...............')
    test(testLoader, covNet)
