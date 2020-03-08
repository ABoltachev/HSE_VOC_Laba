from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.voc_dataset import VocClassificationDataset


def main():
    data_train = VocClassificationDataset('dataset/data', 'train', transform=transforms.ToTensor())
    data_test = VocClassificationDataset('dataset/data', 'val', transform=transforms.ToTensor())
    train_loader = DataLoader(data_train, num_workers=4)
    test_loader = DataLoader(data_test, num_workers=4)

    i = 0
    for x, y in train_loader:
        print(x)
        print(x.size())
        print(y)
        i += 1
        if i == 3:
            break

    print('Test')


if __name__ == '__main__':
    main()
