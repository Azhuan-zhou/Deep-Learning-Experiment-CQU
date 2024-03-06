import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, output_size),
        )

    def forward(self, x):
        return self.model(x)

# 加载训练好的模型权重
model = FNN(input_size=784, hidden_size1=512, hidden_size2=256, hidden_size3=128, hidden_size4=64, output_size=10)
checkpoint = torch.load('model/Relu_0.001.ckpt')
model.load_state_dict(checkpoint)
model.eval()
# 预处理输入数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28))
])

image = cv2.imread('./1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = transform(image).view(-1, 784)  # 转换为张量并拉平成一维向量

# 使用模型进行预测
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)  # 在第一个维度上找到最大值
    predicted_classes = predicted.tolist()  # 将张量转换为Python列表

for i, pred_class in enumerate(predicted_classes):
    print(f'Sample {i + 1}: Predicted class: {pred_class}')