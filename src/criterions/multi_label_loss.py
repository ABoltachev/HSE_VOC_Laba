import torch


class MultiLabelLoss(torch.nn.Module):
    """
    BCEWithLogitsLoss wrapper for multi label classifier
    """
    def __init__(self, num_classes: int, reduction: str = 'mean') -> None:
        super(MultiLabelLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        new_x = x.new_empty((x.size(0), self.num_classes))
        for i in range(self.num_classes):
            new_x[:, i] = x[:, i, 1]
        return self.criterion(new_x, target)
