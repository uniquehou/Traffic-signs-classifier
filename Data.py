from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


TRAIN_DIR, TEST_DIR = 'traffic-sign/train', 'traffic-sign/test'
TRAIN_DOC = 'traffic-sign/train_label.csv'
TEST_DOC = 'traffic-sign/test_label.csv'

class MyDataset(Dataset):
    def __init__(self, dirFile, transform=None):
        # self.data = pd.read_csv(dirFile, nrows=30)
        self.data = pd.read_csv(dirFile)
        self.transform = transform

    def __len__(self):
        return self.data.index.size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration
        image_name, class_id = self.data.loc[idx, ['image_location', 'class_id']]
        image = Image.open(image_name).convert('RGB')
        class_id = int(class_id)
        origin_size = torch.IntTensor(image.size)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'class': class_id, 'filename': os.path.join(*image_name.split('/')[-2:]), 'origin_size': origin_size}
        return sample


def loadData(args):
    train_transform = transforms.Compose([transforms.Resize((args.train_size, args.train_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.Resize((args.train_size, args.train_size)),
                                         transforms.ToTensor()])
    train_dataset = MyDataset(TRAIN_DOC, transform=train_transform)
    test_dataset = MyDataset(TEST_DOC, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    data_loader = {'train': train_loader, 'test': test_loader}
    return data_loader


def visualize(data_loader):
    dataset = data_loader['train'].dataset
    # random display random_size sample of train
    random_size = 3
    for idx in np.random.randint(0, len(dataset), random_size):
        sample = dataset[idx]
        image, class_id, filename = sample['image'], sample['class'], sample['filename']
        plt.title(f"{filename} - {class_id}")
        plt.imshow(transforms.ToPILImage()(image))
        plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Train')
    parse.add_argument('--train-size', type=int, default=128)  # default 128
    parse.add_argument('--train-batch-size', type=int, default=128)  # default 128
    parse.add_argument('--test-batch-size', type=int, default=30)  # default 30
    args = parse.parse_args()
    data_loader = loadData(args)
    visualize(data_loader)
    import torch
    train_class_num, test_class_num = torch.FloatTensor([0]*62), torch.FloatTensor([0]*62)
    for sample in data_loader['train'].dataset:
        train_class_num[sample['class']] += 1
    for sample in data_loader['test'].dataset:
        test_class_num[sample['class']] += 1
    train_loss_weight = torch.mean(train_class_num) / train_class_num
    test_loss_weight = torch.mean(test_class_num) / test_class_num

