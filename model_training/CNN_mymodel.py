import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


'''
输入图像的HWC 32x32x3
'''

BATCH_SIZE = 8
dataset_path='D:\deeplearning\data'

def get_dataset():
    train_dataset = CIFAR10(root=dataset_path, train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root=dataset_path, train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

    
class ImageModel(nn.Module):

    # 搭建网络
    def __init__(self):
        super().__init__()

        # 输入尺寸 32x32x3 

        # 卷积层+池化层
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # (32-3+2*0)/1+1=30 --->30*30*6
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 30/2=15   ---->15*15*6
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1,padding=0)
        # (15-3+2*0)/1+1=13 --->13*13*16
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 13/2=6    ---->6*6*16

        # 全连接层
        # 需要把之前的输出展平  6*6*16=576
        self.linear1=nn.Linear(in_features=576,out_features=120)
        self.linear2=nn.Linear(in_features=120,out_features=84)
        self.linear3=nn.Linear(in_features=84,out_features=10)


    # 前向传播
    def forward(self,x):
        # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.pool1(x)

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.pool2(x)

        # 参1: 样本数(行数), 参2: 列数(特征数), -1表示自动计算.
        x = x.reshape(x.size(0), -1)  # 8行576列
        x=self.linear1(x)
        x=torch.relu(x)

        x=self.linear2(x)
        x=torch.relu(x)

        output=self.linear3(x)
        return output


def train(model,train_dataset):

    dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    # 定义损失函数
    criterion=nn.CrossEntropyLoss()
    # 定义优化器
    optimizer=optim.Adam(model.parameters(),lr=0.01)

    epochs=20

    for epoch_idx in range(epochs):
        # 定义统计训练指标
        total_loss=0.0
        total_samples=0
        total_correct=0
        start =time.time()
        
        # 开始训练
        for x,y in dataloader:
            if(use_cuda):
                x=x.cuda()
                y=y.cuda()

            model.train()
            y_predict=model(x)
            # 计算损失
            loss=criterion(y_predict,y)
            # 梯度清零、反向传播、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            total_correct += (torch.argmax(y_predict, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)  # [第1批总损失 +  第2批总损失 +  第3批总损失 +  ...]
            total_samples += len(y)

        # 一轮训练完毕, 打印该轮的训练信息.
        print(f'epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.2f}, time:{time.time() - start:.2f}s')
    torch.save(model.state_dict(), './model/image_model.pth')
    return 


def evaluate(test_dataset):
    # 1. 创建测试集 数据加载器.
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 2. 创建模型对象.
    model = ImageModel()
    # 3. 加载模型参数.
    model.load_state_dict(torch.load('./model/image_model.pth'))    # pickle文件
    # 4. 定义变量统计 预测正确的样本个数, 总样本个数.
    total_correct, total_samples = 0, 0
    # 5. 遍历数据加载器, 获取到 每批次 的数据.
    for x, y in dataloader:
        # 5.1 切换模型模式.
        model.eval()
        # 5.2 模型预测.
        y_pred = model(x)
        # 5.3 因为训练的时候用了CrossEntropyLoss, 所以搭建神经网络时没有加softmax()激活函数, 这里要用 argmax()来模拟.
        # argmax()函数功能:  返回最大值对应的索引, 充当 -> 该图片的 预测分类.
        y_pred = torch.argmax(y_pred, dim=-1)   # -1 这里表示行.
        # 5.4 统计预测正确的样本个数.
        total_correct += (y_pred == y).sum()
        # 5.5 统计总样本个数.
        total_samples += len(y)

    # 6. 打印正确率(预测结果).
    print(f'Acc: {total_correct / total_samples:.2f}')


def choose_cuda(flag):

    if(flag):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if(device == torch.device('cuda')): 
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    use_cuda=choose_cuda(True)

    train_dataset, test_dataset = get_dataset()
    print(f'use_cuda:{use_cuda}')
    if(use_cuda):
        device='cuda'
        model=ImageModel().cuda()
    else:
        device='cpu'
        model=ImageModel()

    summary(model,input_size=(3,32,32),batch_size=BATCH_SIZE,device=device)
    train(model,train_dataset)
