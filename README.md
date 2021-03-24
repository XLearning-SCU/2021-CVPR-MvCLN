# 2021-CVPR-MvCLN
This repo contains the code and data of the following paper accepted by [CVPR 2021](http://cvpr2021.thecvf.com/)   

[Partially View-aligned Representation Learning with Noise-robust Contrastive Loss](http://pengxi.me/wp-content/uploads/2021/03/2021CVPR-MvCLNwith-supp.pdf)

<img src="https://github.com/XLearning-SCU/2021-CVPR-MvCLN/blob/main/figs/framework.png"  width="897" height="400" />

## Requirements

pytorch==1.5.0 

numpy>=1.18.2

scikit-learn>=0.22.2

munkres>=1.1.2

logging>=0.5.1.2

## Configuration

The hyper-parameters, the training options (including the ratiao of positive to negative, aligned proportions and switch time) are defined in the args. part in run.py.

## Datasets

The Scene-15 and Reuters-dim10 datasets are placed in "datasets" folder. The NoisyMNIST and Caltech101 datasets could be downloaded from [cloud]().

## Usage

After setting the configuration and downloading datasets from the cloud desk, one could run the following code to verify our method on NoisyMNIST-30000 dataset for clustering task.

```bash
python run.py --data 3
```

The expected outputs are as follows:

```bash
******** Training begin, use RobustLoss: 1.0*m, use gpu 0, batch_size = 1024, unaligned_prop = 0.5, NetSeed = 64, DivSeed = 249 ********
=======> Train epoch: 0/80
margin = 5
distance: pos. = 2.5, neg. = 2.5, true neg. = 2.5, false neg. = 2.49
loss = 3.41, epoch_time = 12.07 s
******** testing ********
CAR=0.1012, kmeans: acc=0.1791, nmi=0.0435, ari=0.021
******* neg_dist_mean >= 1.0 * margin, start using fine loss at epoch: 3 *******
=======> Train epoch: 10/80
distance: pos. = 0.76, neg. = 5.38, true neg. = 5.83, false neg. = 1.34
loss = 0.09, epoch_time = 15.17 s
******** testing ********
CAR=0.8712, kmeans: acc=0.9462, nmi=0.8705, ari=0.8862
......
=======> Train epoch: 80/80
distance: pos. = 0.25, neg. = 5.34, true neg. = 5.8, false neg. = 1.17
loss = 0.03, epoch_time = 14.18 s
******** testing ********
CAR=0.8753, kmeans: acc=0.9459, nmi=0.8744, ari=0.8859
******** End, training time = 1276.29 s ********
```

## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{yang2021MvCLN,
   title={Partially View-aligned Representation Learning with Noise-robust Contrastive Loss},
   author={Mouxing Yang, Yunfan Li, Zhenyu Huang, Zitao Liu, Peng Hu, Xi Peng},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month={June},
   year={2021}
}
```

