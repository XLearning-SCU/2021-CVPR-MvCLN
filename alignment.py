import torch
import numpy as np


def tiny_infer(model, device, all_loader):
    model.eval()
    align_out0 = []
    align_out1 = []
    class_labels_cluster = []
    len_alldata = len(all_loader.dataset)
    align_labels = torch.zeros(len_alldata)
    with torch.no_grad():
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1) in enumerate(all_loader):
            test_num = len(labels)

            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            x0 = x0.view(x0.size()[0], -1)
            x1 = x1.view(x1.size()[0], -1)
            h0, h1 = model(x0, x1)

            C = euclidean_dist(h0, h1)
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                C[:, idx[0]] = float("inf")
                align_out0.append((h0[i, :].cpu()).numpy())
                align_out1.append((h1[idx[0], :].cpu()).numpy())
                if class_labels0[i] == class_labels1[idx[0]]:
                    align_labels[1024 * batch_idx + i] = 1

            class_labels_cluster.extend(class_labels0.numpy())

    count = torch.sum(align_labels)
    inference_acc = count.item() / len_alldata

    return np.array(align_out0), np.array(align_out1), np.array(class_labels_cluster), inference_acc


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
