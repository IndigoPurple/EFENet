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
    echo "Done"
