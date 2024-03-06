from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')


# ----------------搭建数据集-----------------
class DataSet(Dataset):
    # 通过下标索引可以访问训练集的所有数据
    def __getitem__(self, index):
        img, label = self.training_set[index]
        return img, label

    def __init__(self, root):
        transform = transforms.Compose(
            [transforms.Resize((28, 28)),
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
                                              batch_size=100,
                                              shuffle=True)
        self.testing_set_loader = DataLoader(self.testing_set,
                                             batch_size=100,
                                             shuffle=True)


# --------------通过tensorboard将训练集的数据可视化---------------
def show():
    dataSet = DataSet(root)
    writer = SummaryWriter('./logs')
    for i in range(len(dataSet.training_set)):
        img, label = dataSet[i]
        writer.add_image(str(label),
                         img,
                         i
                         )
        if i == 100:
            break
    writer.close()


# --------------搭建网络---------------
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            # 输入层到隐藏层1
            nn.Linear(input_size, hidden_size1),
            nn.Sigmoid(),
            # 隐藏1到隐藏层2
            nn.Linear(hidden_size1, hidden_size2),
            nn.Sigmoid(),
            # 隐藏层2到隐藏层3
            nn.Linear(hidden_size2, hidden_size3),
            nn.Sigmoid(),
            # 隐藏层3到隐藏层4
            nn.Linear(hidden_size3, hidden_size4),
            nn.Sigmoid(),
            # 隐藏层4到输出层
            nn.Linear(hidden_size4, output_size),
        )

    def forward(self, x):
        output = self.model(x)
        return output


def drew_loss_accuracy(loss, accuracy, epochs, lr):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = [i + 1 for i in range(epochs)]
    ax1.plot(x, loss, 'm.-.', label='loss', linewidth=1)
    ax1.set_title('Sigmoid, learning rate = {}'.format(lr))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.plot(x, accuracy, 'm.-.', label='accuracy', linewidth=1)
    ax2.set_title('Sigmoid, learning rate = {}'.format(lr))
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy(%)')
    plt.savefig('./picture/Sigmoid_{}.png'.format(lr))


# ----------------训练-----------------------
def training(model, training_set, epochs, learning_rate):
    # ----------GPU计算-----------
    device = torch.device('cuda')
    # -----------实例化损失函数-----------
    loss_function = nn.CrossEntropyLoss().to(device)
    # -------------实例化优化器--------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = len(training_set)  # 一个epoch的训练次数
    # ----------开始训练------------
    lossInEpoch = []
    accuracyInEpoch = []
    for epoch in range(epochs):
        lossNum = 0
        for j, data in enumerate(training_set):
            images, labels = data
            # 连接GPU
            images = images.reshape(-1, 28 * 28).to(device)  # ?
            labels = labels.long().to(device)
            # Forward
            outputs = model.forward(images)
            loss = loss_function(outputs, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 显示训练过程
            if (j + 1) % 100 == 0:
                print('Epoch[{}/{}],Step[{}/{}],Loss:{:.4f}'
                      .format(epoch + 1, epochs, j + 1, steps, loss.item()))
            lossNum += loss.item()
        # 计算一个epoch的平均损失
        lossNum = lossNum / steps
        print('loss in epoch {}: {}'.format(epoch + 1, lossNum))
        # 保存一个epoch的损失
        lossInEpoch.append(lossNum)
        # 测试一个epoch
        testing(testing_s, model, accuracyInEpoch)
    drew_loss_accuracy(lossInEpoch, accuracyInEpoch, epochs, learning_rate)


# -----------testing-------------
def testing(testing_set, model, accuracy):
    # 连接GPU
    device = torch.device('cuda')
    correct = 0
    total = 0
    for images, labels in testing_set:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    accuracy.append(100 * correct / total)


if __name__ == '__main__':
    # ------------设置参数-------------
    root = '../data'
    input_s = 28 * 28
    hidden_1 = 512
    hidden_2 = 256
    hidden_3 = 128
    hidden_4 = 64
    output_s = 10
    num_epochs = 10
    lr_list = [i / 1000 for i in range(1, 6)]
    # -----------------加载数据集------------------
    dataSet = DataSet(root)
    # -----------可视化数据集----------
    # show()
    training_s = dataSet.training_set_loader
    testing_s = dataSet.testing_set_loader
    device = torch.device('cuda')
    # ----------实例化模型-----------
    Model = FNN(input_s, hidden_1, hidden_2, hidden_3, hidden_4, output_s).to(device)
    for lr in lr_list:
        print("---------------learning rate:{}-----------------".format(lr))
        # -----------训练-------------
        training(Model, training_s, num_epochs, lr)
        # -------------保存------------
        torch.save(Model.state_dict(), './model/sigmoid_{}.ckpt'.format(lr))
