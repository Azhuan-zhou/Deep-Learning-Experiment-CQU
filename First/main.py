import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ---------加载数据-----------
class DataSet(Dataset):
    def __getitem__(self, index):
        """empty"""
        pass

    def __init__(self, root):
        self.bath_size_train = 30
        self.bath_size_test = 100
        self.train_dataset = datasets.MNIST(root=root,
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize((28, 28)),
                                                transforms.ToTensor(),
                                            ])
                                            )
        self.test_dataset = datasets.MNIST(root=root,
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.Resize((28, 28)),
                                               transforms.ToTensor(),
                                           ])
                                           )
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.bath_size_train,
                                       shuffle=True
                                       )
        self.test_loader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.bath_size_test,
                                      shuffle=False)


# ---------------查看数据集的信息-------------------
def test(DS: DataSet):
    examples = enumerate(DS.train_loader)
    _, (data, labels) = next(examples)
    print(labels)
    print(data.shape)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth:{}".format(labels[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()


class Regression(nn.Module):

    def __init__(self, input_size, hidden, num_classes):
        """
        模型初始化
        :param input_size: 输入层神经元个数
        :param hidden: 隐藏层个数
        :param num_classes: 输出层神经元个数
        """
        super(Regression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, image):
        x = self.linear1(image)
        out = self.linear2(x)
        return out


# ----------------训练模型---------------
def Train(DS: DataSet, input_size, num_classes, learning_rate, num_epoch, batch_size):
    """
    初始化，训练模型
    :param DS: 数据集
    :param input_size: 输入特征尺寸
    :param num_classes: 分类类别个数
    :param learning_rate: 学习率
    :param num_epoch: 迭代次数
    :param batch_size: 一个批次的数据量
    :return:
    """
    # ----------初始化-------------
    if torch.cuda.is_available():
        model = Regression(input_size=input_size, hidden=50, num_classes=num_classes).cuda()
    else:
        model = Regression(input_size=input_size, hidden=50, num_classes=num_classes)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # -----------训练---------------
    model.train()
    for epoch in range(num_epoch):
        for i, (image, label) in enumerate(DS.train_loader):
            if torch.cuda.is_available():
                image = Variable(image.view(-1, 28 * 28)).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image.view(-1, 28 * 28))
                label = Variable(label)

            # --------------forward+Backward+Optimize------------
            optimizer.zero_grad()  # 将上轮梯度置零
            outputs = model.forward(image)  # forward
            _, Predicted = torch.max(outputs.data, 1)  # 返回一行中最大的元素和索引，即predicted
            loss = criterion(outputs, label)  # 计算损失
            loss.backward()  # backward
            optimizer.step()  # Optimize, 迭代优化器，更新参数
            # ---------------每迭代100次，显示当前损失----------
            if (i + 1) % 100 == 0:
                print("Epoch:{}/{},Step:[{}/{}],Loss:{}".format(epoch + 1, num_epoch, i + 1,
                                                                len(DS.train_dataset) // batch_size, loss.item()))
    Test(DS.test_loader, model)


# -------------测试集计算正确率-----------------
def Test(test_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        count = 0
        for image, label in test_loader:
            if torch.cuda.is_available():
                image = image.reshape(-1, 28 * 28).cuda()
                label = label.cuda()
            else:
                image = image.reshape(-1, 28 * 28)
            outputs = model.forward(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
            count += 1
        print('Accuracy of the model on the {} test images: {} %'.format(count * 1000, 100 * correct / total))


if __name__ == '__main__':
    dataset = DataSet('../data')
    test(dataset)
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.01
    num_epoch = 5
    batch_size = 30
    Train(dataset, input_size, num_classes, learning_rate, num_epoch, batch_size)
