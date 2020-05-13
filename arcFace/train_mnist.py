import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


# feature_dim=512,s=30,m=0.5

class ArcLoss(nn.Module):
    def __init__(self, feature_dim, cls_dim):
        super(ArcLoss, self).__init__()
        self.weight = nn.Parameter(torch.randn(feature_dim, cls_dim))  # feature*weight -> n*cls_dim
        self.s = 10 # 防止梯度爆炸
        self.m = 0.1 # 改变角度

    def forward(self, feature):
        feature = F.normalize(feature, dim=1) # 2范数归一化
        w = F.normalize(self.weight, dim=0) # ２范数归一化

        cos_theat = torch.matmul(feature, w) / 10 # 计算角度
        a = torch.acos(cos_theat) # 反三角计算角度

        top = torch.exp((torch.cos(a + self.m)) * self.s) # 计算改变的角度
        _top = torch.exp((torch.cos(a)) * self.s) # 计算为原的角度

        bottom = torch.sum(torch.exp(cos_theat * self.s), dim=1).view(-1, 1) # 计算分母

        divide = (top / (bottom - _top + top)) + 1e-10 # 更新公式

        return divide


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(64 * 2 * 2, 2),
            # nn.Linear(128,2),
        )

        self.out_layer = nn.Linear(2, 10)

    def forward(self, x):
        cnn_out = self.cnn_layer(x)
        x = torch.reshape(cnn_out, (-1, 64 * 2 * 2))
        feature_out = self.feature_layer(x)
        out = F.log_softmax(self.out_layer(feature_out), dim=1)
        return feature_out, out


def draw_img(feature, targets, epoch, save_path="images"):
    if os.path.isdir(save_path) != True:
        os.makedirs(save_path)

    color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()

    for j in cls:
        # mask如果targets == j就为True
        mask = [targets == j]
        # 找到类别为j的特征点
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right")  # 如果写在plot上面，则标签内容不能显示完整
        plt.title("epoch={}".format(str(epoch)))

    plt.savefig('{}/{}.jpg'.format(save_path, epoch + 1))
    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    class_num = 10

    net = Net().cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST('mnist', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    arcloss = ArcLoss(2, 10).cuda()
    clsloss = nn.NLLLoss(reduction='sum').cuda()

    # # #分类的优化器
    # cls_opt=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9,weight_decay=0.0005)
    # # #衰减系数，用来调整opt的学习率，
    # scheduler=optim.lr_scheduler.StepLR(cls_opt,step_size=20,gamma=0.8)
    cls_opt = optim.Adam(net.parameters())

    arcloss_opt = optim.Adam(arcloss.parameters())


    for epoch in range(100):
        feat = []
        label = []
        for j, (input, target) in enumerate(dataloader):
            input = input.cuda()
            target1 = target.cuda()
            target2 = torch.zeros(target.size(0), 10).scatter(1, target.view(-1, 1), 1).cuda()
            feature_out, out = net(input)

            value = torch.argmax(out, dim=1)

            arc_loss = torch.log(arcloss(feature_out))

            cls_loss = clsloss(out, target1)
            arcface_loss = clsloss(arc_loss, target1)

            loss = cls_loss + arcface_loss

            cls_opt.zero_grad()
            arcloss_opt.zero_grad()
            loss.backward()
            cls_opt.step()
            arcloss_opt.step()

            feat.append(feature_out)
            label.append(target)

            if j % 200 == 0:
                print(f'epochs--{epoch}--{j}/{len(dataloader)},loss:{loss}')

        features = torch.cat(feat, dim=0)
        labels = torch.cat(label, dim=0)
        draw_img(features.data.cpu(), labels.data.cpu(), epoch)