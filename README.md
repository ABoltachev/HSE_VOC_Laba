# HSE training project (VOC)

Neural Network model for multi-classification. Deep learning engine - PyTorch<br>


## Clone repository without models

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ABoltachev/HSE_VOC_Laba.git<br>
To clone model:
 * sudo apt-get install git-lfs
 * git lfs clone

## Train

To run train need download [dataset](https://mega.nz/file/VN9lgRYB#IzDxlIzpjjUQTmtlgon5FPnn23PKfRvMfeXOA-4xezk) to data folder<br>
Run script: python3 -m src.train --dataset_path=dataset/data --tensorboard_path=tensorboard --checkpoint_path=exp --log_path=log --exp_name=resnet18_new_augm --use_multi_label_classifier=True --npz_filepath=dataset/ms_gray.npz --use_gpu=True --batch_size=16 --momentum=0.9 --lr=5e-3 --num_epochs=40


## Test

Run script: python3 -m src.test --dataset_path=test_images --model_path=model/model.best.ap


## ResNet18
### Result
|    Model version   | Loss (Train) | Average precision (Train) | Loss (Valid) | Average precision (Valid) |
|:------------------:|:------------:|:-------------------------:|:------------:|:-------------------------:|
| ResNet18 version 3 | 2.6351       | 0.7615                    | 3.5190       | 0.6551                    |

### Version 3
Dataset with augmentation and normalization<br>
Images in a gray scale<br>
An unprepared model was used<br>

|    Hyperparameter   |  Values |
|:-------------------:|:-------:|
| Image Size          | 300x300 |
| Batch size          | 16      |
| Accumulate gradient | 1       |
| Epochs              | 40      |
| Optimizer           | SGD     |
| Momentum            | 0.0     |
| Learning rate       | 1e-3    |

[TensorBoard](https://tensorboard.dev/experiment/cPElRsr3SgiD1ObzQU9i2w/)

Train loss:<br>
[Train loss log](results/run-resnet18_multi_label_classifier-tag-train_loss.csv)<br>
![ResNet18 v1](results/train_loss.svg)
Train ap:<br>
[Train AP log](results/run-resnet18_multi_label_classifier-tag-train_ap.csv)<br>
![ResNet18 v1](results/train_ap.svg)
Valid loss:<br>
[Valid loss log](results/run-resnet18_multi_label_classifier-tag-valid_loss.csv)<br>
![ResNet18 v1](results/valid_loss.svg)
Valid ap:<br>
[Valid AP log](results/run-resnet18_multi_label_classifier-tag-valid_ap.csv)<br>
![ResNet18 v1](results/valid_ap.svg)
