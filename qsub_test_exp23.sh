#!/bin/bash

#block(name=exp23_test, threads=5, memory=5000, subtasks=1, gpu=true, hours=100)
python -u train_crossnetpp_vimeo.py  \
--dataset MPII   \
--mode test \
--display 100 \
--batch_size 1  \
--step_size 50000 \
--gamma 0.1 \
--loss CharbonnierLoss \
--optim Adam \
--lr 0.00001  \
--checkpoints_dir ./checkpoints_exp23_test/ \
--frame_num 41 \
--checkpoint_file ./checkpoints_exp15/CP135000.pth \
--with_GAN_loss 1 \
--img_save_path result/exp23_test \
--net_type multiflowfusion2 \
--pretrained 0 \
--gpu_id 0
    echo "Done" 

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
