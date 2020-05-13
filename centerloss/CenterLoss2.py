# *_*coding:utf-8 *_*
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# torch.manual_seed(0)

class CenterLoss(nn.Module):
    def __init__(self, cls_num, featur_num):
        super().__init__()

        self.cls_num = cls_num
        self.featur_num = featur_num
        self.center = nn.Parameter(t.randn(cls_num, featur_num))

    def forward(self, xs, ys):  # xs=feature, ys=target
        # xs = Variable(xs, requires_grad=True)
        # xs = F.normalize(xs)
        center_exp = self.center.index_select(dim=0, index=ys.long()) # 拿到类别的中心点
        counts = self.center.new_ones(self.center.size(0)) # 选择类别数量
        ones = self.center.new_ones(ys.size(0)) # 拿到标签的数量
        counts = counts.scatter_add_(0, ys.long(), ones) # 所有类别标签在一个批次的直方图
        count_dis = counts.index_select(dim=0, index=ys.long())
        loss = t.sum(t.sum((xs - center_exp).pow(2), dim=1) / 2.0 / count_dis.float())
        return loss

if __name__ == '__main__':
    print('-'*80)
    ct = CenterLoss(10, 2)
    y = torch.Tensor([0,0,2,1])
    feat = torch.zeros(4,2)
    print(list(ct.parameters()))
    out = ct(feat, y)
    print(out.item())