import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2

class GigaDataset(Dataset):
    ''' Giga Dataset '''

    def __init__(self, data_path_lr, data_path_hr, frame_num=30, transform=None, is_train=True):
        self.is_train = is_train
        ## file list
        self.data_path_lr = data_path_lr
        self.data_path_hr = data_path_hr
        self.folder_list = self.get_list()
        self.file_list = ['im_%04d.jpg' % i for i in range(1, frame_num+1)]
        # transform
        self.transform = transform
        self.crop = None
        self.frame_num = frame_num  # frame number
        # self.N = self.M * self.K    # total number
        if is_train:
            self.M = 8 * 16  # sequences number
        else:
            self.M = 8 * 2
        self.N = self.M

    def get_list(self):
        folder_list = []
        if self.is_train:
            for i in range(1, 9):
                for j in range(1, 17):
                    folder_list.append('%04d/%02d/' % (i, j))
        else:
            for i in range(1, 9):
                for j in range(17, 19):
                    folder_list.append('%04d/%02d/' % (i, j))
        return folder_list

    def get_frame(self, m, f, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # print (len(self.file_list),f)
        if mode == 'HR':
            filename = self.data_path_hr + self.folder_list[m] + self.file_list[f]
        elif mode == 'LR':
            filename = self.data_path_lr + self.folder_list[m] + self.file_list[f]
        image = Image.open(filename)
        return image

    def get_all_frames(self, m, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # print (len(self.file_list),f)
        # print('get all frames>>>>>>>>>>')
        if mode == 'HR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.data_path_hr + self.folder_list[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        elif mode == 'LR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.data_path_lr + self.folder_list[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        return images

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        m = idx
        f_st = 0

        sample = dict()

        sample['input_img1_LR'] = self.get_frame(m, f_st, mode='LR')
        if self.transform:
            sample['input_img1_LR'] = self.transform(sample['input_img1_LR'])

        sample['input_img1_HR'] = self.get_frame(m, f_st, mode='HR')
        if self.transform:
            sample['input_img1_HR'] = self.transform(sample['input_img1_HR'])

        # sample['input_img1_SR'] = self.get_frame(m, f_st, mode='SR')
        # if self.transform:
        #     sample['input_img1_SR'] = self.transform(sample['input_img1_SR'])

        sample['input_img2_HR'] = self.get_frame(m, self.frame_num-1, mode='HR')
        if self.transform:
            sample['input_img2_HR'] = self.transform(sample['input_img2_HR'])
        sample['input_LR'] = self.get_all_frames(m, mode='LR')
        if self.transform:
            sample['input_LR'] = self.transform(sample['input_LR'])
        sample['input_HR'] = self.get_all_frames(m, mode='HR')
        if self.transform:
            sample['input_HR'] = self.transform(sample['input_HR'])

        seq_id = self.folder_list[m]
        if self.is_train:
            return sample
        else:
            return sample, seq_id

if __name__ == "__main__":
    data_path_corrupted = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_noise/'
    data_path_MDSR = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_upsampled_MDSR/'
    data_path_clean = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences/'
    data_list_file = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sep_trainlist.txt'
    # composed = transforms.Compose([transforms.RandomCrop((128,128)),
    #                                 transforms.ToTensor()])

    composed = transforms.Compose([transforms.ToTensor()])
    dataset = VimeoDataset(data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, frame_window_size=2,
                           transform=composed)

    #### test pytorch dataset
    # print(len(dataset))

    # fig = plt.figure()
    # plt.axis('off')
    # plt.ioff()
    # im = plt.imshow(np.zeros((dataset.H, dataset.W, 3)), vmin=0, vmax=1)

    # for i in range(len(dataset)-1, 0, -1):
    #     sample = dataset[i]
    #     for t in sample:
    #         print(t, sample[t].size())
    #         im.set_data(sample[t].numpy().transpose(1,2,0))
    #         plt.pause(0.1)
    # exit()

    #### test dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)
    print(len(dataset), len(dataloader))

    import cPickle as pickle
    import os

    img_name = ['img1_LR', 'img1_SR', 'img1_HR', 'img2_HR']
    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch, sample_batched['gt'].size())
        # visualization
        images_batch = sample_batched['input_img1_LR']
        batch_size = images_batch.size()[0]
        im_size = images_batch.size()[1:]

        print(i_batch)
        save_dir = './vimeo_sr/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open(save_dir + str(i_batch + 1), 'wb')
        pickle.dump(sample_batched, f)
        f.close()

        # print(batch_size, im_size)
        # grid = utils.make_grid(images_batch, nrow=2)
        # plt.imshow(grid.numpy().transpose(1,2,0))
        # plt.show()

        # observe 4th batch and stop.
        # if i_batch == 3:
        #     plt.figure()
        #     show_landmarks_batch(sample_batched)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     break
