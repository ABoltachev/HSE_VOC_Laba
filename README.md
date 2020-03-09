# HSE training project (VOC)

Neural Network model for multi-classification. Deep learning engine - PyTorch<br>

## Results
|    Model version   | Loss (Train) | Average precision (Train) | Loss (Valid) | Average precision (Valid) |
|:------------------:|:------------:|:-------------------------:|:------------:|:-------------------------:|
| [ResNet18 version 1](#resnet18_v1) | 3.1240       | 0.6809                    | 3.7812       | 0.5852                    |

## ResNet18
### Version 1  <a name="resnet18_v1"></a>
Dataset without augmentation and normalization<br>
An unprepared model was used<br>

|    Hyperparameter   |  Values |
|:-------------------:|:-------:|
| Image Size          | 300x300 |
| Batch size          | 16      |
| Accumulate gradient | 1       |
| Epochs              | 40      |
| Optimizer           | SGD     |
| Momentum            | 0.9     |
| Learning rate       | 5e-3    |

Train loss:<br>
[Train loss log](resnet18/v1/results/run-resnet18-tag-train_loss.csv)<br>
![ResNet18 v1](resnet18/v1/results/train_loss.svg)
Train ap:<br>
[Train AP log](resnet18/v1/results/run-resnet18-tag-train_ap.csv)<br>
![ResNet18 v1](resnet18/v1/results/train_ap.svg)
Valid loss:<br>
[Valid loss log](resnet18/v1/results/run-resnet18-tag-valid_loss.csv)<br>
![ResNet18 v1](resnet18/v1/results/valid_loss.svg)
Valid ap:<br>
[Valid AP log](resnet18/v1/results/run-resnet18-tag-valid_ap.csv)<br>
![ResNet18 v1](resnet18/v1/results/valid_ap.svg)
