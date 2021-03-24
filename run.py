import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader
from models import *
from alignment import tiny_infer
from Clustering import Clustering

parser = argparse.ArgumentParser(description='MvCLN in PyTorch')
parser.add_argument('--data', default='0', type=int,
                    help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='80', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float, help='learning rate of adam')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels')
parser.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-ap', '--aligned-prop', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx to use.')
parser.add_argument('-r', '--robust', default=1, type=int, help='use our robust loss or not')
parser.add_argument('-t', '--switching-time', default=1.0, type=float, help='start fine when neg_dist>=t*margin')
parser.add_argument('-s', '--start-fine', default=False, type=bool, help='flag to start use robust loss or not')
# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []


class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, use_robust_loss, args):
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        if use_robust_loss == 1:
            if args.start_fine:
                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, 0.5) * (margin - pair_dist), min=0.0), 2)
            else:
                loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss


def train(train_loader, model, criterion, optimizer, epoch, args):
    pos_dist = 0  # mean distance of pos. pairs
    neg_dist = 0
    false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
    true_neg_dist = 0
    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0

    if epoch % 10 == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    loss_value = 0
    for batch_idx, (x0, x1, labels, real_labels) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        x0, x1, labels, real_labels = x0.to(args.gpu), x1.to(args.gpu), labels.to(args.gpu), real_labels.to(args.gpu)
        try:
            h0, h1 = model(x0.view(x0.size()[0], -1), x1.view(x1.size()[0], -1))
        except:
            print("error raise in batch", batch_idx)

        pair_dist = F.pairwise_distance(h0, h1)  # use Euclidean distance to measure similarity
        pos_dist += torch.sum(pair_dist[labels == 1])
        neg_dist += torch.sum(pair_dist[labels == 0])
        true_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])
        pos_count += len(pair_dist[labels == 1])
        neg_count += len(pair_dist[labels == 0])
        true_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])

        loss = criterion(pair_dist, labels, args.margin, args.robust, args)
        loss_value += loss.item()
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0

    pos_dist /= pos_count
    neg_dist /= neg_count
    true_neg_dist /= true_neg_count
    false_neg_dist /= false_neg_count
    if epoch != 0 and args.robust == 1 and neg_dist >= args.switching_time * args.margin and not args.start_fine:
        # start fine when the mean distance of neg. pairs is greater than switching_time * margin
        args.start_fine = True
        logging.info("******* neg_dist_mean >= {} * margin, start using fine loss at epoch: {} *******".format(
            args.switching_time, epoch + 1))

    # margin = the pos. distance + neg. distance before training
    if epoch == 0 and args.margin != 1.0:
        args.margin = max(1, round((pos_dist + neg_dist).item()))
        logging.info("margin = {}".format(args.margin))

    if epoch % 10 == 0:
        logging.info("distance: pos. = {}, neg. = {}, true neg. = {}, false neg. = {}".format(round(pos_dist.item(), 2),
                                                                                              round(neg_dist.item(), 2),
                                                                                              round(
                                                                                                  true_neg_dist.item(),
                                                                                                  2), round(
                false_neg_dist.item(), 2)))
        logging.info(
            "loss = {}, epoch_time = {} s".format(round(loss_value / len(train_loader), 2), round(epoch_time, 2)))

    return pos_dist, neg_dist, false_neg_dist, true_neg_dist, epoch_time


def plot(acc, nmi, ari, CAR, args, data_name):
    x = range(0, args.epochs + 1, 1)
    fig_clustering = plt.figure()
    ax_clustering = fig_clustering.add_subplot(1, 1, 1)
    ax_clustering.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        args.robust * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_clustering.plot(x, acc, label='acc')
    lns2 = ax_clustering.plot(x, ari, label='ari')
    lns3 = ax_clustering.plot(x, nmi, label='nmi')
    lns4 = ax_clustering.plot(x, CAR, label='CAR')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax_clustering.legend(lns, labs, loc=0)
    ax_clustering.grid()
    ax_clustering.set_xlabel("epoch(s)")
    ax_clustering.plot()

    fig_dist = plt.figure()
    ax_dist_mean = fig_dist.add_subplot(1, 1, 1)
    ax_dist_mean.set_title(data_name + ", " + "Noise=" + str(args.noisy_training) + ", RobustLoss=" + str(
        args.robust * args.switching_time) + ", neg_prop=" + str(args.neg_prop))
    lns1 = ax_dist_mean.plot(x, pos_dist_mean_list, label='pos. dist')
    lns2 = ax_dist_mean.plot(x, neg_dist_mean_list, label='neg. dist')
    lns3 = ax_dist_mean.plot(x, false_neg_dist_mean_list, label='false neg. dist')
    lns4 = ax_dist_mean.plot(x, true_neg_dist_mean_list, label='true neg. dist')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax_dist_mean.legend(lns, labs, loc=0)
    ax_dist_mean.grid()
    ax_dist_mean.set_xlabel("epoch(s)")
    plt.show()


def main():
    args = parser.parse_args()
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000']
    NetSeed = 64
    # random.seed(NetSeed)
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(NetSeed)  # 为当前GPU设置随机种子

    train_pair_loader, all_loader, divide_seed = loader(args.batch_size, args.neg_prop, args.aligned_prop,
                                                        args.noisy_training, data_name[args.data])
    if args.data == 0:
        model = MvCLNfcScene().to(args.gpu)
    elif args.data == 1:
        model = MvCLNfcCaltech().to(args.gpu)
    elif args.data == 2:
        model = MvCLNfcReuters().to(args.gpu)
    elif args.data == 3:
        model = MvCLNfcMNIST().to(args.gpu)

    criterion = NoiseRobustLoss().to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    path = os.path.join("./log/" + str(data_name[args.data]) + "_" + 'time=' + time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                             time.localtime(
                                                                                                 time.time())))
    os.mkdir(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path + '.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(
        "******** Training begin, use RobustLoss: {}*m, use gpu {}, batch_size = {}, unaligned_prop = {}, NetSeed = {}, DivSeed = {} ********".format(
            args.robust * args.switching_time, args.gpu, args.batch_size, (1 - args.aligned_prop), NetSeed,
            divide_seed))

    CAR_list = []
    acc_list, nmi_list, ari_list = [], [], []
    train_time = 0
    # train
    for i in range(0, args.epochs + 1):
        if i == 0:
            with torch.no_grad():
                pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time = train(
                    train_pair_loader, model, criterion, optimizer, i, args)
        else:
            pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time = train(train_pair_loader,
                                                                                                      model, criterion,
                                                                                                      optimizer, i,
                                                                                                      args)
        train_time += epoch_time
        pos_dist_mean_list.append(pos_dist_mean.item())
        neg_dist_mean_list.append(neg_dist_mean.item())
        true_neg_dist_mean_list.append(true_neg_dist_mean.item())
        false_neg_dist_mean_list.append(false_neg_dist_mean.item())

        # test
        v0, v1, pred_label, alignment_rate = tiny_infer(model, args.gpu, all_loader)
        CAR_list.append(alignment_rate)
        data = []
        data.append(v0)
        data.append(v1)
        y_pred, ret = Clustering(data, pred_label)
        if i % 10 == 0:
            logging.info("******** testing ********")
            logging.info(
                "CAR={}, kmeans: acc={}, nmi={}, ari={}".format(round(alignment_rate, 4), ret['kmeans']['accuracy'],
                                                                ret['kmeans']['NMI'], ret['kmeans']['ARI']))
        acc_list.append(ret['kmeans']['accuracy'])
        nmi_list.append(ret['kmeans']['NMI'])
        ari_list.append(ret['kmeans']['ARI'])

    # plot(acc_list, nmi_list, ari_list, CAR_list, args, data_name[args.data])
    logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))


if __name__ == '__main__':
    main()
