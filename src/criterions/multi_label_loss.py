import torch


class MultiLabelLoss(torch.nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super(MultiLabelLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, x, target):
        new_x = x.new_empty((x.size(0), self.num_classes))
        for i in range(self.num_classes):
            new_x[:, i] = x[:, i, 1]
        return self.criterion(new_x, target)
