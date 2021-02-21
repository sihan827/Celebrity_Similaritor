import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'


def get_dataset_mean_std(data_path):
    """
    make a simple DataSet from data_path and calculate mean and std of RGB channels
    :param data_path: a data path which is a directory that separated by classes
    :return: mean and std of dataset
    """
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2)

    mean = 0.
    std = 0.

    for imgs, _ in loader:
        imgs.to(DEVICE)
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    # mean=[0.5893, 0.4750, 0.4330], std=[0.2573, 0.2273, 0.2134] for celebrities_face
    # mean=[0.5961, 0.4563, 0.3906], std=[0.2184, 0.1944, 0.1852] for utk_face
    # mean=[0.6361, 0.4875, 0.4189], std=[0.2105, 0.1893, 0.1820] for face_age
    return mean, std


def get_dataset(data_path, mean, std, target_transform=None):
    """
    make a dataset normalized by mean and std from data path
    :param data_path:a data path which is a directory that separated by classes
    :param mean: mean value to normalize
    :param std: std value to normalize
    :param target_transform: use to ImageFolder's target_transform, default None
    :return: transformed DataSet
    """
    # can add more data augmentation
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomHorizontalFlip()
    ])

    return datasets.ImageFolder(root=data_path, transform=preprocess, target_transform=target_transform)


def check_dataset(dataset):
    """
    show images of one batch(16) in dataset
    :param dataset: DataSet that you want to check
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    imgs_grid = make_grid(images, padding=0)
    np_grid = imgs_grid.numpy()
    plt.figure(figsize=(10, 7))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    for i in labels:
        print(dataset.classes[i.item()])
    plt.show()


def train_test_set_split(dataset, dataset_name, test_size=0.1):
    """
    split DataSet to train and test by Subset
    if indices txt files exist, indices will be loaded based on txt files
    otherwise, split indices by train_test_split and save indices in txt files
    train and test set will be separated by above indices
    :param dataset: base DataSet
    :param dataset_name: name of dataset
    :param test_size: test size rate you want to split from dataset, default 0.1
    :return: train set, test set, train set labels
    """
    train_indices_path = './' + dataset_name + '_train_indices(' + str(test_size) + ').txt'
    test_indices_path = './' + dataset_name + '_test_indices(' + str(test_size) + ').txt'
    try:
        train_indices = []
        test_indices = []
        file = open(train_indices_path, 'rt', encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            train_indices.append(int(line[:-1]))
        file.close()
        file = open(test_indices_path, 'rt', encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            test_indices.append(int(line[:-1]))
        file.close()
        train_labels = [dataset.targets[i] for i in train_indices]
    except FileNotFoundError:
        indices = np.arange(len(dataset))
        labels = np.array(dataset.targets)
        train_indices, test_indices, train_labels, _ = train_test_split(
            indices, labels, test_size=test_size, stratify=labels
        )
        file = open(train_indices_path, 'wt', encoding='utf-8')
        for i in train_indices:
            line = str(i) + '\n'
            file.write(line)
        file.close()
        file = open(test_indices_path, 'wt', encoding='utf-8')
        for i in test_indices:
            line = str(i) + '\n'
            file.write(line)
        file.close()

    train_set = torch.utils.data.Subset(dataset, indices=train_indices)
    test_set = torch.utils.data.Subset(dataset, indices=test_indices)
    return train_set, test_set, train_labels


def train_valid_loader_split(train_set, train_labels, batch_size=32, valid_size=0.1, stratify=True):
    """
    split train set to train sampler and valid sampler using SubsetRandomSampler
    :param train_set: base train set
    :param train_labels: base train label(can be used as stratify
    :param batch_size: loader's batch size
    :param valid_size: valid size rate you want to split from train set, default 0.1
    :param stratify: stratify split option, default True
    :return: train loader, valid loader
    """
    indices = np.arange(len(train_set))
    if stratify:
        train_indices, valid_indices, _, _ = train_test_split(
            indices, train_labels, test_size=valid_size, stratify=train_labels
        )
    else:
        train_indices, valid_indices, _, _ = train_test_split(
            indices, train_labels, test_size=valid_size
        )
    train_sampler = SubsetRandomSampler(indices=train_indices)
    valid_sampler = SubsetRandomSampler(indices=valid_indices)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size, sampler=train_sampler, num_workers=2
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size, sampler=valid_sampler, num_workers=2
    )
    return train_loader, valid_loader


def calculate_class_weight(dataset):
    """
    calculate weight for each class
    used for calculating weighted loss
    :param dataset: full dataset you want to calculate weight for each class
    :return: tensor of weight for each class
    """
    weight_list = []
    for i in range(len(dataset.classes)):
        n = dataset.targets.count(i)
        weight = torch.tensor([len(dataset) / n])
        weight_list.append(weight)
    weight_tensor = torch.cat(weight_list, dim=0)
    max_val = torch.max(weight_tensor, dim=0).values.item()
    return torch.div(weight_tensor, max_val)


def face_target(label):
    """
    this function is for Facial Age dataset in Kaggle uploaded by Fazle Rabbi
    downloaded from->https://www.kaggle.com/frabbisw/facial-age
    this function transforms target label value to real age value so can be used in regression model
    you must check Facial Age dataset's targets and classes first and modify this function to fit in your dataset
    I modified my Facial Age dataset so age 100 class is included in age 99
    :param label: parameter for callable target_transform function
    :return: transformed target value
    """
    classes = []
    for i in range(1, 97):
        classes.append(torch.tensor([i], dtype=torch.float32))
    classes.append(torch.tensor([99], dtype=torch.float32))
    return classes[label]


if __name__ == '__main__':
    dataset = get_dataset(
        './celebrities_face',
        [0.5893, 0.4750, 0.4330],
        [0.2573, 0.2273, 0.2134],
    )
    check_dataset(dataset)
    train_set, test_set, train_labels = train_test_set_split(dataset, 'celebrities_face', test_size=0.1)
    train_loader, valid_loader = train_valid_loader_split(train_set, train_labels, batch_size=32, valid_size=0.1)
    daitater = iter(train_loader)
    X, y = daitater.next()
    print(y.shape)
    print(len(train_set))
    print(len(test_set))

