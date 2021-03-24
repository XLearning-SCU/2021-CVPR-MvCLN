import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import TT_split, normalize
import torch
import random


def load_data(dataset, neg_prop, test_prop, is_noise):
    all_data = []
    train_pairs = []
    label = []

    mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'NoisyMNIST-30000':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    divide_seed = random.randint(1, 1000)  #
    train_idx, test_idx = TT_split(len(label), test_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]

    # Use test_prop*sizeof(all data) to train the MvCLN, and shuffle the rest data to simulate the unaligned data.
    # Note that, MvCLN establishes the correspondence of the all data rather than the unaligned portion in the testing.
    # When test_prop = 0, MvCLN is directly performed on the all data without shuffling.
    if test_prop != 0:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
        all_data.append(np.concatenate((train_X, test_X)).T)
        all_data.append(np.concatenate((train_Y, test_Y)).T)
        all_label = np.concatenate((train_label, test_label))
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))
    elif test_prop == 0:
        all_data.append(train_X.T)
        all_data.append(train_Y.T)
        all_label, all_label_X, all_label_Y = train_label, train_label, train_label

    # pair construction. view 0 and 1 refer to pairs constructed for training. noisy and real labels refer to 0/1 label of those pairs
    view0, view1, noisy_labels, real_labels, _, _ = get_pairs(train_X, train_Y, neg_prop, train_label)

    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1
    print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X)), 2))

    if is_noise == 0:  # training with real_labels, v/t with real_labels
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
    else:  # training with labels, v/t with real_labels
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
    train_pairs.append(view0.T)
    train_pairs.append(view1.T)
    train_pair_real_labels = real_labels

    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, divide_seed


def get_pairs(train_X, train_Y, neg_prop, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])
    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)
        for k in range(neg_prop):
            view0.append(train_X[j])
            view1.append(train_Y[neg_idx[k]])
            labels.append(0)
            class_labels0.append(train_label[j])
            class_labels1.append(train_label[neg_idx[k]])
            if train_label[j] != train_label[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


class GetDataset(Dataset):
    def __init__(self, data, labels, real_labels):
        self.data = data
        self.labels = labels
        self.real_labels = real_labels

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        real_label = np.int64(self.real_labels[index])
        return fea0, fea1, label, real_label

    def __len__(self):
        return len(self.labels)


class GetAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), torch.from_numpy(self.data[1][:, index]).float()
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        return fea0, fea1, label, class_labels0, class_labels1

    def __len__(self):
        return len(self.labels)


def loader(train_bs, neg_prop, test_prop, is_noise, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param test_prop: known aligned proportions for training MvCLN
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param data_idx: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    divide_seed = load_data(dataset, neg_prop, test_prop, is_noise)
    train_pair_dataset = GetDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    all_dataset = GetAllDataset(all_data, all_label, all_label_X, all_label_Y)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=1024,
        shuffle=True
    )
    return train_pair_loader, all_loader, divide_seed
