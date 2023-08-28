# EFENet: Reference-based Video Super-Resolution with Enhanced Flow Estimation
In this repository we provide code of the paper:
> **EFENet: Reference-based Video Super-Resolution with Enhanced Flow Estimation**

> Yaping Zhao, Mengqi Ji, Ruqi Huang, Bin Wang, Shengjin Wang

> arxiv link: http://arxiv.org/abs/2110.07797

<p align="center">
<img src="img/teaser.png">
</p>

# Usage
0. For pre-requisites, run:
```
conda env create -f environment.yml
conda activate efenet
```
1. Pretrained model is currently available at [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/zhaoyp_connect_hku_hk/EZISUcLG-iRFvrEwPIEKlvMBAZgsDVz7yj91CwovqsBWBw?e=0W6ZA2) and [Baidu Netdisk](https://pan.baidu.com/s/1BeAKAENf_TPPuUr-oRzajw ) (password: efen), download the `CP10000.pth` and put it in the `pretrained` folder. **This pretrained model only uses one training sample for demo purpose. If you want to train your own model, please prepare your own training set.** 

2. For EFENet training, run:
```
sh train.sh
```
or
```
python train_efenet_vimeo.py  \
--dataset demo   \
--display 100 \
--batch_size 1  \
--step_size 50000 \
--gamma 0.1 \
--loss CharbonnierLoss \
--optim Adam \
--lr 0.00001  \
--checkpoints_dir ./checkpoints/ \
--frame_num 7 \
--checkpoint_file ./pretrained/CP10000.pth \
--with_GAN_loss 0 \
--img_save_path result/ \
--net_type multiflowfusion5 \
--pretrained 1 \
--gpu_id 0 
```
3. For EFENet testing, run:
```
sh test.sh
```
or
```
python train_efenet_vimeo.py  \
--dataset demo   \
--mode test \
--display 100 \
--batch_size 1  \
--step_size 50000 \
--gamma 0.1 \
--loss CharbonnierLoss \
--optim Adam \
--lr 0.00001  \
--checkpoints_dir ./checkpoints/ \
--frame_num 7 \
--checkpoint_file ./pretrained/CP10000.pth \
--with_GAN_loss 0 \
--img_save_path result/ \
--net_type multiflowfusion5 \
--pretrained 0 \
--gpu_id 0
```

4. If positive, you will get models in the `checkpoints/` folder when training and results in the `result/` folder when testing.

# Dataset
Dataset is stored in the folder `dataset/`, where subfolders `clean/`, `corrupted/`, `SISR/` contain ground truth HR images, corrupted LR images, upsampled LR images by interpolation (e.g., bicubic) or SISR methods.
Images in `SISR/` could be as same as in `corrupted/`, though preprocessing by advanced SISR methods (e.g., MDSR) brings a small performance boost.

`testlist.txt` and `trainlist.txt` could be modified for your experiment on other datasets. 

This repo only provides a sample for demo purposes. 

# Citation
Cite our paper if you find it interesting!
```
@misc{zhao2021efenet,
      title={EFENet: Reference-based Video Super-Resolution with Enhanced Flow Estimation}, 
      author={Yaping Zhao and Mengqi Ji and Ruqi Huang and Bin Wang and Shengjin Wang},
      year={2021},
      eprint={2110.07797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
