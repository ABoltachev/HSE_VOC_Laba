# TODO: add augmentation
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision.models import resnet18
from tqdm import tqdm

from dataset.voc_dataset import CLASSES
from dataset.voc_dataset import VocClassificationDataset
from src.utils import init_logging, get_ap_score, save_model, calculate_mean_std, DATETIME


def main():
    parser = argparse.ArgumentParser(description='Train scrip')
    parser.add_argument('--dataset_path',
                        required=True,
                        type=str,
                        help='Root directory of the VOC Dataset')
    parser.add_argument('--tensorboard_path',
                        required=True,
                        type=str,
                        help='TensorBoard logs path')
    parser.add_argument('--checkpoint_path',
                        required=True,
                        type=str,
                        help='Model checkpoints path')
    parser.add_argument('--log_path',
                        default=None,
                        type=str,
                        help='Logs path')
    parser.add_argument('--tensorboard_update_step',
                        default=10,
                        type=int,
                        help='Update TensorBoard logs every n step')
    parser.add_argument('--exp_name',
                        default=None,
                        type=str,
                        help='The name of the expration. If not set, then the current date is used by expration name')
    parser.add_argument('--use_gpu',
                        default=False,
                        type=bool,
                        help='When this flag is set, the GPU is used for training')
    # Train parameters
    train_args_parser = parser.add_argument_group('Train parameters')
    train_args_parser.add_argument('--npz_filepath',
                                   default=None,
                                   type=str,
                                   help='Path to npz file with mean and std for normalization')
    train_args_parser.add_argument('--image_size',
                                   nargs='+',
                                   default=[300, 300],
                                   type=int,
                                   help='Desired output size (default: (300, 300))'
                                   )
    train_args_parser.add_argument('--accum_grad',
                                   default=1,
                                   type=int,
                                   help='Gradient accumulation')
    train_args_parser.add_argument('--num_epochs',
                                   default=90,
                                   type=int,
                                   help='Number of epochs')
    train_args_parser.add_argument('--batch_size',
                                   default=32,
                                   type=int,
                                   help='Batch size')
    # Optimization algorithm parameters
    opt_args_parser = parser.add_argument_group('Optimization algorithm parameters')
    opt_args_parser.add_argument('--optim',
                                 default='adam',
                                 type=str,
                                 choices=['adam', 'sgd'],
                                 help="Optimization algorithm [adam or sgd]")
    opt_args_parser.add_argument('--lr',
                                 default=1e-3,
                                 type=float,
                                 help='Learning rate')
    opt_args_parser.add_argument('--weight_decay',
                                 default=0,
                                 type=float,
                                 help='L2 penalty')
    # Adam parameters
    adam_args_parser = parser.add_argument_group('Adam parameters')
    adam_args_parser.add_argument('--betas',
                                  nargs='+',
                                  default=[0.9, 0.999],
                                  type=float,
                                  help='Coefficients used for computing running '
                                       'averages of gradient and its square (default: (0.9, 0.999))')
    # SGD parameters
    sgd_args_parser = parser.add_argument_group('SGD parameters')
    sgd_args_parser.add_argument('--momentum',
                                 default=0,
                                 type=float,
                                 help='Momentum factor')
    args = parser.parse_args()

    train_args_dict = {a.dest: getattr(args, a.dest, None) for a in train_args_parser._group_actions}
    opt_args_dict = {a.dest: getattr(args, a.dest, None) for a in opt_args_parser._group_actions}

    os.makedirs(args.log_path, exist_ok=True)
    init_logging(args.log_path, args.exp_name)

    seed = torch.initial_seed()
    logging.info(f'Used seed : {seed}')

    if args.exp_name:
        tensorboard_path = os.path.join(args.tensorboard_path, args.exp_name)
    else:
        tensorboard_path = os.path.join(args.tensorboard_path, DATETIME)
    tb_writer = SummaryWriter(log_dir=tensorboard_path)
    logging.info('TensorBoardX summary writer created')

    device = torch.device('cuda' if args.use_gpu else 'cpu')
    logging.info(f'Train device: {device}')

    if args.npz_filepath:
        npzfile = np.load(args.npz_filepath)
        mean = npzfile['mean']
        std = npzfile['std']
    else:
        transforms_for_msc = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                                transforms.ToTensor()])
        data_for_msc = VocClassificationDataset(args.dataset_path, 'train', transform=transforms_for_msc)
        loader_for_msc = data.DataLoader(data_for_msc, num_workers=4)

        mean, std = calculate_mean_std(loader_for_msc, tuple(args.image_size), device)

    logging.info(f'mean: {mean}')
    logging.info(f'std: {std}')

    train_transforms = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                           transforms.Grayscale(num_output_channels=3),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    valid_transforms = transforms.Compose([transforms.Resize(tuple(args.image_size)),
                                           transforms.Grayscale(num_output_channels=3),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])

    train_data = VocClassificationDataset(args.dataset_path, 'train', transform=train_transforms)
    train_data_loader = data.DataLoader(
        train_data,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        batch_size=args.batch_size)
    logging.info('Train data loader created')
    valid_data = VocClassificationDataset(args.dataset_path, 'val', transform=valid_transforms)
    valid_data_loader = data.DataLoader(
        valid_data,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        batch_size=args.batch_size)
    logging.info('Valid data loader created')

    model = resnet18(num_classes=len(CLASSES))
    model = model.to(device)
    logging.info('model created')

    if args.optim == 'adam':
        opt_args_dict['betas'] = tuple(args.betas)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=tuple(args.betas), weight_decay=args.weight_decay)
        logging.info('Adam optimizer created')
    elif args.optim == 'sgd':
        opt_args_dict['momentum'] = args.momentum
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        logging.info('SGD optimizer created')
    else:
        logging.error('Unknown optimizer')
        raise ValueError(f"Unknown optimizer: '{args.optim}'")

    # TODO: Use optim.lr_scheduler

    logging.info('Starting train...')

    steps_per_epoch = len(train_data) // args.batch_size
    total_steps = steps_per_epoch * args.num_epochs

    logging.info(f'Total steps: {total_steps}')
    logging.info(f'Steps per epoch: {steps_per_epoch}')

    done_steps = 1
    best_loss = float('+inf')
    best_ap = float('0')

    if args.exp_name:
        checkpoint_path = os.path.join(args.checkpoint_path, args.exp_name)
    else:
        checkpoint_path = os.path.join(args.checkpoint_path, DATETIME)
    os.makedirs(checkpoint_path, exist_ok=True)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    m = torch.nn.Sigmoid()

    for epoch in range(args.num_epochs):
        cur_train_loss = .0
        cur_train_ap = .0
        model.train()

        tbar = tqdm(train_data_loader, dynamic_ncols=True)
        for i, (imgs, targets) in enumerate(tbar):
            if done_steps % args.accum_grad == 0:
                optimizer.zero_grad()
                logging.debug('Zero gradient')

            imgs, targets = imgs.to(device), targets.to(device)

            out = model(imgs)
            logging.debug(f'out: {out.size()}')
            logging.debug(f'targets: {targets.size()}')
            loss = criterion(out, targets)

            loss.backward()
            optimizer.step()

            ap = get_ap_score(targets.cpu().detach().numpy(), m(out).cpu().detach().numpy())
            cur_train_loss += loss.item()
            cur_train_ap += ap

            tbar.set_description_str(f'Train: {epoch + 1 } '
                                     f'Loss: {cur_train_loss / float((i + 1) * args.batch_size):.4f} '
                                     f'AP: {cur_train_ap / float((i + 1) * args.batch_size):.4f}')

            if done_steps % args.tensorboard_update_step == 0:

                tb_writer.add_scalar('train/loss', cur_train_loss / float((i + 1) * args.batch_size), done_steps)
                tb_writer.add_scalar('train/ap', cur_train_ap / float((i + 1) * args.batch_size), done_steps)

            done_steps += 1

        cur_train_loss = cur_train_loss / len(train_data)
        cur_train_ap = cur_train_ap / len(train_data)

        logging.info(f'Epoch: {epoch + 1}\tStep: {done_steps}\t'
                     f'Loss: {cur_train_loss:.4f}\tAP: {cur_train_ap:.4f}')

        test_loss = 0
        test_ap = 0
        model.eval()

        with torch.no_grad():
            dev_tbar = tqdm(valid_data_loader, dynamic_ncols=True)
            for i, (dev_imgs, dev_targets) in enumerate(dev_tbar):
                dev_imgs, dev_targets = dev_imgs.to(device), dev_targets.to(device)

                dev_out = model(dev_imgs)

                test_loss += criterion(dev_out, dev_targets).item()

                test_ap += get_ap_score(dev_targets.cpu().detach().numpy(), m(dev_out).cpu().detach().numpy())

                dev_tbar.set_description(f'Valid: {epoch + 1} '
                                         f'Loss: {test_loss / float((i + 1) * args.batch_size):.4f} '
                                         f'AP: {test_ap / float((i + 1) * args.batch_size):.4f}')

        test_ap = test_ap / len(valid_data)
        test_loss = test_loss / len(valid_data)

        logging.info(f'Epoch: {epoch + 1}\tStep: {done_steps}\t'
                     f'Valid/Loss: {test_loss:.4f}\tValid/AP: {test_ap:.4f}')

        tb_writer.add_scalar('valid/loss', test_loss, done_steps)
        tb_writer.add_scalar('valid/ap', test_ap, done_steps)

        if best_loss - test_loss > 1e-6:
            best_loss = test_loss
            best_loss_checkpoint_path = os.path.join(checkpoint_path, f'model.best.loss')
            save_model(best_loss_checkpoint_path, epoch + 1, done_steps, optimizer, model, seed,
                       train_transforms, valid_transforms,
                       train_args_dict, opt_args_dict)

        if best_ap - test_ap < -1e-6:
            best_ap = test_ap
            best_ap_checkpoint_path = os.path.join(checkpoint_path, f'model.best.ap')
            save_model(best_ap_checkpoint_path, epoch + 1, done_steps, optimizer, model, seed,
                       train_transforms, valid_transforms,
                       train_args_dict, opt_args_dict)

        epoch_checkpoint_path = os.path.join(checkpoint_path, f'model.{epoch + 1}')
        save_model(epoch_checkpoint_path, epoch + 1, done_steps, optimizer, model, seed,
                   train_transforms, valid_transforms,
                   train_args_dict, opt_args_dict)


if __name__ == '__main__':
    main()
