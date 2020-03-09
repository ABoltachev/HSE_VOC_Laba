# HSE training project (VOC)

Neural Network model for multi-classification. Deep learning engine - PyTorch

## ResNet18
### Version 1
Dataset without augmentation and normalization<br>

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
![ResNet18 v1](models/resnet18_not_augment_not_pretained/train_loss.svg)
Train ap:<br>
![ResNet18 v1](models/resnet18_not_augment_not_pretained/train_ap.svg)
Valid loss:<br>
![ResNet18 v1](models/resnet18_not_augment_not_pretained/valid_loss.svg)
Valid ap:<br>
![ResNet18 v1](models/resnet18_not_augment_not_pretained/valid_ap.svg)
