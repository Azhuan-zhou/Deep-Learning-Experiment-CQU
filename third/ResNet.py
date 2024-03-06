from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.models import resnet18
from Covolution_MINST import train, test, show, draw


class MyResNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(MyResNet, self).__init__()
        self.inplanes = 64
        device = torch.device('cuda')
        resNet = resnet18().to(device)
        num_ftrs = resNet.fc.in_features
        # 改变输入通道
        resNet.conv1 = nn.Conv2d(input_size, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # 改变输出通道
        resNet.fc = nn.Linear(num_ftrs, output_size)
        resNet = resNet.to(device)
        self.model = resNet

    def forward(self, x):
        return self.model(x)


# ----------------搭建数据集-----------------
class DataSet(Dataset):
    # 通过下标索引可以访问训练集的所有数据
    def __getitem__(self, index):
        dataset = self.training_set + self.training_set
        img, label = dataset[index]
        return img, label

    def __init__(self, root, batch_size):
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
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


if __name__ == '__main__':
    # -----------参数初始化-----------
    inputChanel = 1
    outputSize = 10
    epochs = 30
    learningRate = 0.01
    batch_size = 18
    device = torch.device('cuda')

    print('1.Loading data............')
    dataset = DataSet('../data', batch_size)
    # 训练数据
    trainLoader = dataset.training_set_loader
    # 测试数据
    testLoader = dataset.testing_set_loader
    # 加载模型
    resnet = MyResNet().to(device)

    print('2.Training............')
    lossInEpoch = []
    train(resnet, trainLoader, epochs, learningRate, lossInEpoch)
    draw(lossInEpoch, epochs)

    print('3.testing...............')
    test(testLoader, resnet)
