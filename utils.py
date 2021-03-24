import numpy as np
import random
import logging
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x


def random_index(n_all, n_train, seed):
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_idx = random_idx[0:n_train]
    test_idx = random_idx[n_train:n_all]
    return train_idx, test_idx


def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(np.int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(np.int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx


def initLogging(logFilename):
    # 日志格式化方式
    LOG_FORMAT = "%(asctime)s\tFile \"%(filename)s\",LINE %(lineno)-4d : %(levelname)-8s %(message)s"
    # 日期格式化方式
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=logFilename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter = logging.Formatter(LOG_FORMAT);
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(formatter);
    logging.getLogger('').addHandler(console);


def svm_classify(data, label, test_prop, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    seed = random.randint(0, 1000)
    train_idx, test_idx = TT_split(data.shape[1], test_prop, seed)
    train_data = np.concatenate([data[0][train_idx], data[1][train_idx]], axis=1)
    test_data = np.concatenate([data[0][test_idx], data[1][test_idx]], axis=1)
    test_label = label[test_idx]
    train_label = label[train_idx]

    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    return test_acc


def knn(data, label, test_prop, k):
    seed = random.randint(0, 1000)
    train_idx, test_idx = TT_split(data.shape[1], test_prop, seed)
    train_data = np.concatenate([data[0][train_idx], data[1][train_idx]], axis=1)
    test_data = np.concatenate([data[0][test_idx], data[1][test_idx]], axis=1)
    test_label = label[test_idx]
    train_label = label[train_idx]

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_data, train_label)
    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    return test_acc


