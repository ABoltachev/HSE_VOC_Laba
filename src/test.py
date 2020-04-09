import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image

from dataset.voc_dataset import CLASSES
from src.utils import init_logging, load_model


def main():
    parser = argparse.ArgumentParser(description='Train scrip')
    parser.add_argument('--dataset_path',
                        required=True,
                        type=str,
                        help='Root directory of the VOC Dataset')
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='Path to trained model')
    parser.add_argument('--log_path',
                        default='test_logs',
                        type=str,
                        help='Logs path')
    parser.add_argument('--exp_name',
                        default=None,
                        type=str,
                        help='The name of the expration. If not set, then the current date is used by expration name')
    parser.add_argument('--use_gpu',
                        default=False,
                        type=bool,
                        help='When this flag is set, the GPU is used for training')

    args = parser.parse_args()

    os.makedirs(args.log_path, exist_ok=True)
    init_logging(args.log_path, args.exp_name)

    device = torch.device('cuda' if args.use_gpu else 'cpu')

    logging.info(f'Test device: {device}')

    model, image_transforms = load_model(args.model_path, device=device)

    logging.info('Model loaded')

    m = torch.nn.Sigmoid()

    logging.info('Start test')

    np_classes = np.array(CLASSES)

    for image_path in glob(f'{args.dataset_path}/*.jpg'):
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = image_transforms(image).unsqueeze(0)
        inputs.to(device)
        out = model(inputs)
        targets_scores = m(out).cpu().detach().squeeze(0)
        classes_probabilities = targets_scores.softmax(1)
        targets = classes_probabilities[:, 1] > 0.5
        index_classes = targets.nonzero().squeeze().numpy()
        classes = np_classes[index_classes]
        true_probabilities = classes_probabilities[index_classes, 1].numpy()
        if isinstance(classes, np.ndarray):
            result = [f"{classes[i]}({true_probabilities[i]:.2f})" for i in range(len(classes))]
        else:
            result = f"{classes}({true_probabilities:.2f})"
        logging.info(f'Image name: {image_name},\timage classes: {result}')


if __name__ == '__main__':
    main()
