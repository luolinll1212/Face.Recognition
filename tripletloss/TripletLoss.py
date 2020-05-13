# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, embed, margin=0.2):
        super(TripletLoss, self).__init__()
        self.embed = embed
        self.margin = margin

    def forward(self, feat):
        # 取anchor,positive,negative
        anchor = feat[:, 0:self.embed]
        positive = feat[:, self.embed:self.embed*2]
        negative = feat[:, self.embed*2:]
        # 计算距离
        pos_dist = torch.sum(torch.pow(anchor-positive, 2), dim=1).unsqueeze(1)
        neg_dist = torch.sum(torch.pow(anchor-negative, 2), dim=1).unsqueeze(1)
        # 计算损失
        basic_loss = pos_dist - neg_dist + self.margin
        # 取最大的数，全部为负取0,与np.maximum函数一样
        loss = F.relu(basic_loss, inplace=True).max() # 返回最大值

        return loss

if __name__ == '__main__':

    torch.manual_seed(0)
    embed = 128
    batch = 10
    tripletloss = TripletLoss(embed, margin=0.2)
    feat = torch.randn(batch,embed*3)
    out = tripletloss(feat)
    print(out.item())
