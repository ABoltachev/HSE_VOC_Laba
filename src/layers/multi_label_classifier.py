import torch
import torch.nn as nn


class MultiLabelClassifier(torch.nn.Module):
    """
    Module List of n binary classifiers, where n is number classes
    """
    def __init__(self, input_features: int, num_classes: int) -> None:
        super(MultiLabelClassifier, self).__init__()
        self.num_classes = num_classes
        self.binary_classifiers = nn.ModuleList([nn.Linear(input_features, 2) for i in range(num_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = x.new_empty((x.size(0), self.num_classes, 2), requires_grad=True)
        for i, classifier in enumerate(self.binary_classifiers):
            outputs[:, i, :] = classifier(x)

        return outputs
