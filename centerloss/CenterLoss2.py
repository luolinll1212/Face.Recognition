# *_*coding:utf-8 *_*
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CenterLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, feat, labels):
        """
        x: [batch_size, feature_len]
        labels: [bacth_size]
        """
        loss = 0.0
        for i, g in enumerate(labels):
            loss += torch.norm(feat[i] - self.centers[g.int(),:]) ** 2
        loss = loss / feat.size(0) / 2
        return loss.sum()


if __name__ == '__main__':
    torch.manual_seed(0)
    ct = CenterLoss(10, 2)
    y = torch.Tensor([0,0,2,1])
    feat = torch.zeros(4, 2).requires_grad_()
    print(list(ct.parameters()))
    out = ct(feat, y)
    print(out.item())
    out.backward() # 计算梯度
    print(ct.centers.grad)
    print(feat.grad)
