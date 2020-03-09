import argparse
import datetime
import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from torch.utils import data
from tqdm import tqdm

from dataset.voc_dataset import VocClassificationDataset

__DEBUG_MODE = False
DATETIME = f'{datetime.datetime.now().strftime("%y-%m-%d_%I-%M-%S_%p")}'
__CALCULATE_MEAN_STD = 'msc'


def calculate_mean_std(dataset_loader, image_size, device=torch.device('cpu')):
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


def save_model(name_model, epoch, done_steps,
               optimizer, model, seed,
               train_transforms, valid_transforms,
               train_args_dict, opt_args_dict):
    state = {
        'epoch': epoch,
        'done_steps': done_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'train_transforms': train_transforms,
        'valid_transforms': valid_transforms,
        'train_args_dict': train_args_dict,
        'opt_args_dict': opt_args_dict,
        'seed': seed,
    }
    torch.save(state, name_model)


def get_ap_score(y_true, y_scores):
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


def init_logging(log_path, exp_name):
    # set up logging to file - see previous section for more details
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
    msc_args_group.add_argument('--out_path',
                                required=True,
                                type=str,
                                help=f'Output path for {__CALCULATE_MEAN_STD}')
    args = parser.parse_args()

    if args.algorithm == __CALCULATE_MEAN_STD:
        transforms_for_msc = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                                 transforms.ToTensor()])
        data_for_msc = VocClassificationDataset(args.dataset_path, 'train', transform=transforms_for_msc)
        loader_for_msc = data.DataLoader(data_for_msc, num_workers=4)

        device = torch.device('cuda' if args.use_gpu else 'cpu')
        mean, std = calculate_mean_std(loader_for_msc, tuple(args.image_size), device)
        np.savez(args.out_path, mean=mean, std=std)


if __name__ == '__main__':
    main()
