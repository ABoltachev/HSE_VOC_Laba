import argparse
import datetime
import logging
import os
from typing import Union, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchvision.models import resnet18
from tqdm import tqdm

from dataset.voc_dataset import VocClassificationDataset, CLASSES
from src.layers import MultiLabelClassifier

__DEBUG_MODE = False
DATETIME = f'{datetime.datetime.now().strftime("%y-%m-%d_%I-%M-%S_%p")}'
__CALCULATE_MEAN_STD = 'msc'


def calculate_mean_std(dataset_loader: data.DataLoader, image_size: Tuple[int, int],
                       device: Union[torch.device, str] = torch.device('cpu')) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating mean and std of dataset

    :param dataset_loader: train dataset loader
    :param image_size: image size
    :param device: the device on which the calculations will be performed
    :return: two numpy array: first - mean, second - std
    """
    mean = 0.0
    mean_bar = tqdm(dataset_loader, dynamic_ncols=True)
    mean_bar.set_description('mean calculation')
    for images, _ in mean_bar:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataset_loader.dataset)

    var = 0.0
    std_bar = tqdm(dataset_loader, dynamic_ncols=True)
    std_bar.set_description('std calculation')
    for images, _ in std_bar:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(dataset_loader.dataset) * image_size[0] * image_size[1]))

    return mean.cpu().detach().numpy(), std.cpu().detach().numpy()


def save_model(name_model: str, epoch: int,
               optimizer: Optimizer, model: torch.nn.Module,
               train_transforms: transforms.Compose, valid_transforms: transforms.Compose,
               train_args_dict: dict, opt_args_dict: dict) -> None:
    """
    Save model, optimizer, transforms, train information and dicts arguments for train and optimizer

    :param name_model: path to model file
    :param epoch: epoch number for saved model
    :param optimizer: optimizer for train
    :param model: model to save
    :param train_transforms: transforms for train dataset
    :param valid_transforms: transforms for validation and test dataset
    :param train_args_dict: arguments for train
    :param opt_args_dict: arguments for optimizer
    """
    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'train_transforms': train_transforms,
        'valid_transforms': valid_transforms,
        'train_args_dict': train_args_dict,
        'opt_args_dict': opt_args_dict,
    }
    torch.save(state, name_model)


def load_model(model_path: str, test: bool = True,
               device: Union[torch.device, str] = 'cpu') -> Tuple[torch.nn.Module, transforms.Compose]:
    """
    Load model

    :param model_path: path to model
    :param test: if this flag is set then the model is switched to eval mode
    :param device: device to model
    :return: model and valid transformer
    """
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_state_dict = state['model']
    train_args_dict = state['train_args_dict']
    use_multi_label_classifier = train_args_dict['use_multi_label_classifier'] if train_args_dict.get(
        'use_multi_label_classifier') else False
    image_transforms = state['valid_transforms']

    model = resnet18(num_classes=len(CLASSES))
    if use_multi_label_classifier:
        model.fc = MultiLabelClassifier(512, len(CLASSES))
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    if test:
        assert use_multi_label_classifier, "test mode is not available for this type of model"
        model.eval()

    return model, image_transforms


def get_ap_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculation average precision

    :param y_true: true label
    :param y_scores: score of label
    :return: average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


def init_logging(log_path: str, exp_name: str) -> None:
    """
    Logger initialization

    :param log_path: path to log files
    :param exp_name: log file name
    """
    # set up logging to file - see previous section for more details
    if log_path:
        if exp_name:
            log_filename = os.path.join(log_path, exp_name)
        else:
            log_filename = os.path.join(log_path, DATETIME)
        logging.basicConfig(level=logging.DEBUG if __DEBUG_MODE else logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=f'{log_filename}.log',
                            filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if __DEBUG_MODE else logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def main():
    parser = argparse.ArgumentParser(description='Train scrip')
    parser.add_argument('--algorithm',
                        required=True,
                        type=str,
                        choices=[__CALCULATE_MEAN_STD],
                        help='Name of algorithm')
    msc_args_group = parser.add_argument_group(f'{__CALCULATE_MEAN_STD} arguments')
    msc_args_group.add_argument('--dataset_path',
                                required=True,
                                type=str,
                                help='Root directory of the VOC Dataset')
    msc_args_group.add_argument('--use_gpu',
                                default=False,
                                type=bool,
                                help=f'When this flag is set, the GPU is used for {__CALCULATE_MEAN_STD}')
    msc_args_group.add_argument('--image_size',
                                nargs='+',
                                default=[300, 300],
                                type=int,
                                help='Desired output size (default: (300, 300))')
    msc_args_group.add_argument('--is_gray',
                                default=False,
                                type=bool,
                                help='If set to true, then convert image to grayscale')
    msc_args_group.add_argument('--out_path',
                                required=True,
                                type=str,
                                help=f'Output path for {__CALCULATE_MEAN_STD}')
    args = parser.parse_args()

    if args.algorithm == __CALCULATE_MEAN_STD:
        if args.is_gray:
            transforms_for_msc = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                                     transforms.Grayscale(num_output_channels=3),
                                                     transforms.ToTensor()])
        else:
            transforms_for_msc = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                                     transforms.ToTensor()])
        data_for_msc = VocClassificationDataset(args.dataset_path, 'train', transform=transforms_for_msc)
        loader_for_msc = data.DataLoader(data_for_msc, num_workers=4)

        device = torch.device('cuda' if args.use_gpu else 'cpu')
        mean, std = calculate_mean_std(loader_for_msc, tuple(args.image_size), device)
        np.savez(args.out_path, mean=mean, std=std)


if __name__ == '__main__':
    main()
