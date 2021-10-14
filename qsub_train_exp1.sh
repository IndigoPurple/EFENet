#!/bin/bash

#block(name=exp1, threads=5, memory=5000, subtasks=1, gpu=true, hours=100)
python -u train_crossnet_vimeo_landmark.py  \
--dataset MPII   \
--display 100 --batch_size 1  \
--step_size 50000 --gamma 0.2 \
--loss CharbonnierLoss \
--optim Adam \
--lr 0.00005  \
--checkpoints_dir ./checkpoints_exp1/ \
--frame_window_size 7 \
--checkpoint_file ./checkpoints_exp51/CP350000.pth \
--with_GAN_loss 1
--img_save_path test_exp1_result2
    echo "Done" 

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
