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

The Scene-15 and Reuters-dim10 datasets are placed in "datasets" folder. The NoisyMNIST and Caltech101 datasets could be downloaded from [Google cloud](https://drive.google.com/drive/folders/1WFbxX1X_pNJX0bDRkbF577mRrviIcyKe?usp=sharing) or [Baidu cloud](https://pan.baidu.com/s/1NdgRH3k9Pq9SQjrorWSEeg) with password "rqv4".

## Usage

After setting the configuration and downloading datasets from the cloud desk, one could run the following code to verify our method on NoisyMNIST-30000 dataset for clustering task.

```bash
python run.py --data 3
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

