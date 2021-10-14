import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2


class MpiiDataset(Dataset):
    ''' Mpii Dataset '''

    def __init__(self, data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, min_window_size=3,
                 transform=None, is_train=True, require_seqid=False, frame_num=41):
        # assert frame_window_size%2==1, "frame_window_size should be odd"
        ## file list
        self.data_list_file = data_list_file
        self.folder_list_corrupted, self.frame_num_list = self.get_list(data_path_corrupted)
        self.folder_list_clean, _ = self.get_list(data_path_clean)
        self.folder_list_MDSR, _ = self.get_list(data_path_MDSR)
        self.frame_num = frame_num
        # self.file_list = ['im'+str(i)+'.png' for i in range(1,8)]
        # transform
        self.transform = transform
        self.crop = None
        self.is_train = is_train
        self.M = len(self.folder_list_clean)  # sequences number

        self.require_seqid = require_seqid
        if is_train:
            self.N = self.M
        else:
            self.N = self.M

        self.random_frame_id = np.random.RandomState(100)
        self.random_frame_window = np.random.RandomState(200)
        self.min_window_size = min_window_size

    def get_list(self, data_path):
        folder_list = list()
        frame_num_list = list()
        with open(self.data_list_file, 'r') as f_index:
            reader = csv.reader(f_index)
            for row in reader:
                if row:
                    seq_id, frame_num = row[0].split(' ')
                    folder_list.append(data_path + seq_id + '/')
                    frame_num_list.append(int(frame_num))
        return folder_list, frame_num_list

    def get_frame(self, m, f, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # print (len(self.file_list),f)
        if mode == 'HR':
            filename = self.folder_list_clean[m] + self.file_list[f]
        elif mode == 'LR':
            filename = self.folder_list_corrupted[m] + self.file_list[f]
        elif mode == 'SR':
            filename = self.folder_list_MDSR[m] + self.file_list[f]

        # image = Image.open(filename)
        image = cv2.imread(filename)
        # image = image[:256, :320, :]
        return image

    def get_all_frames(self, m, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # images = None
        # if mode == 'HR':
        #     for f in self.file_list:
        #         filename = self.folder_list_clean[m] + f
        #         image = cv2.imread(filename)
        #         # image = image[:256, :320, :]
        #         if images is None:
        #             images = image
        #         else:
        #             images = np.concatenate((images, image), axis=2)
        #     # print('HR>>>>>>>>')
        #     # print images.shape
        # elif mode == 'LR':
        #     # print('mode: LR')
        #     for f in self.file_list:
        #         filename = self.folder_list_corrupted[m] + f
        #         image = cv2.imread(filename)
        #         # image = image[:256, :512, :]
        #         if images is None:
        #             images = image
        #         else:
        #             images = np.concatenate((images, image), axis=2)
        #     # print('LR>>>>>>>>')
        #     # print images.shape
        # elif mode == 'SR':
        #     for f in self.file_list:
        #         filename = self.folder_list_MDSR[m] + f
        #         image = cv2.imread(filename)
        #         # image = image[:256, :512, :]
        #         if images is None:
        #             images = image
        #         else:
        #             images = np.concatenate((images, image), axis=2)
        # return images
        # print (len(self.file_list),f)
        # images = np.array()
        # if mode == 'HR':
        #     for f in self.file_list:
        #         filename = self.folder_list_clean[m] + f
        #         image = cv2.imread(filename)
        #         images = np.concatenate((images, image), axis=2)
        # elif mode == 'LR':
        #     for f in self.file_list:
        #         filename = self.folder_list_corrupted[m] + f
        #         image = cv2.imread(filename)
        #         images = np.concatenate((images, image), axis=2)
        # elif mode == 'SR':
        #     for f in self.file_list:
        #         filename = self.folder_list_MDSR[m] + f
        #         image = cv2.imread(filename)
        #         images = np.concatenate((images, image), axis=2)
        # print(images.shape)
        # return images
        if mode == 'HR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_clean[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        elif mode == 'LR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_corrupted[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        elif mode == 'SR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_MDSR[m] + f
                image = cv2.imread(filename)
                images = np.concatenate(images_list, axis=2)
            images = np.concatenate(images_list, axis=2)
        return images

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # initialize seed for every sample
        if self.transform:
            seed = np.random.randint(2147483647)

        max_frame_num = self.frame_num_list[idx]

        self.file_list = ['im_%04d.png' % i for i in range(1, max_frame_num + 1)]

        image = self.get_frame(idx, 0)
        H, W, _ = image.shape

        self.H = H
        self.W = W

        m = idx
        f_st = 0

        sample = dict()

        sample['input_img1_LR'] = self.get_frame(m, f_st, mode='LR')
        if self.transform:
            sample['input_img1_LR'] = self.transform(sample['input_img1_LR'])

        sample['input_img1_HR'] = self.get_frame(m, f_st, mode='HR')
        if self.transform:
            sample['input_img1_HR'] = self.transform(sample['input_img1_HR'])

        sample['input_img1_SR'] = self.get_frame(m, f_st, mode='SR')
        if self.transform:
            sample['input_img1_SR'] = self.transform(sample['input_img1_SR'])

        sample['input_img2_HR'] = self.get_frame(m, self.frame_num-1, mode='HR')
        if self.transform:
            sample['input_img2_HR'] = self.transform(sample['input_img2_HR'])
        sample['input_LR'] = self.get_all_frames(m, mode='LR')
        if self.transform:
            sample['input_LR'] = self.transform(sample['input_LR'])
        sample['input_HR'] = self.get_all_frames(m, mode='HR')
        if self.transform:
            sample['input_HR'] = self.transform(sample['input_HR'])

        if self.require_seqid:
            seq_id = self.folder_list_clean[m]
            return sample, seq_id
        else:
            return sample


if __name__ == "__main__":
    print('get new mpii frame>>>>>>')
    data_path_corrupted = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/MPII_1/LR_4x/'
    data_path_MDSR = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/MPII_1/LR_4x/'
    data_path_clean = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/MPII_1/HR/'
    data_list_file = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/MPII_1/MPII_1.txt'
    # composed = transforms.Compose([transforms.RandomCrop((128,128)),
    #                                 transforms.ToTensor()])

    composed = transforms.Compose([transforms.ToTensor()])
    dataset = MpiiDataset(data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, min_window_size=3,
                          transform=composed, is_train=False)

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)
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

        print(i_batch, images_batch.size())
        # save_dir = './vimeo_sr/'
        # if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)
        # f= open(save_dir + str(i_batch + 1),'wb')
        # pickle.dump(sample_batched,f)
        # f.close()

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







# import numpy as np
# import csv
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
# import random
# import cv2
#
# class MpiiDataset(Dataset):
#     ''' Vimeo Dataset '''
#
#     def __init__(self, data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, frame_num=41,
#                  transform=None, is_train=True, require_seqid=False):
#         # assert frame_window_size%2==1, "frame_window_size should be odd"
#         ## file list
#         self.data_list_file = data_list_file
#         # seq_list = []
#         frame_num_list = []
#         with open(data_list_file, 'r') as file_to_read:
#             while True:
#                 lines = file_to_read.readline()
#                 if not lines:
#                     break
#                 s_tmp, f_tmp = [int(i) for i in lines.split()]
#                 # seq_list.append(p_tmp)
#                 frame_num_list.append(f_tmp)
#         self.frame_num_list = np.array(frame_num_list)
#         self.folder_list_corrupted = self.get_list(data_path_corrupted)
#         # self.frame_num_list = self.get_list(data_path_corrupted)
#         self.folder_list_clean = self.get_list(data_path_clean)
#         self.folder_list_MDSR = self.get_list(data_path_MDSR)
#         self.file_list = ['im_%04d.png' % i for i in range(1, frame_num+1)]
#         # transform
#         self.transform = transform
#         self.crop = None
#         self.is_train = is_train
#         self.M = len(self.folder_list_clean)  # sequences number
#         self.frame_num = frame_num  # frame number
#         # self.N = self.M * self.K    # total number
#         if is_train:
#             self.N = self.M * self.frame_num  # total number
#         else:
#             self.N = self.M
#         # get shape
#         self.require_seqid = require_seqid
#         # if self.require_seqid:
#         #     buff, seqid = self.__getitem__(0)
#         #     _, self.H, self.W = buff['input_img1_LR'].size()
#         # else:
#         #     _, self.H, self.W = self.__getitem__(0)['input_img1_LR'].size()
#
#     def get_list(self, data_path):
#         folder_list = list()
#         with open(self.data_list_file, 'r') as f_index:
#             reader = csv.reader(f_index)
#             for row in reader:
#                 if row:
#                     folder_list.append(data_path + row[0].split(' ')[0] + '/')
#         return folder_list
#
#     def get_frame(self, m, f, mode='LR'):
#         ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
#         # print (len(self.file_list),f)
#         if mode == 'HR':
#             filename = self.folder_list_clean[m] + self.file_list[f]
#         elif mode == 'LR':
#             filename = self.folder_list_corrupted[m] + self.file_list[f]
#         elif mode == 'SR':
#             filename = self.folder_list_MDSR[m] + self.file_list[f]
#         image = Image.open(filename)
#         return image
#
#     def get_all_frames(self, m, mode='LR'):
#         ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
#         # print (len(self.file_list),f)
#         # print('get all frames>>>>>>>>>>')
#         images = None
#         if mode == 'HR':
#             images_list = tuple()
#             for i, f in enumerate(self.file_list):
#                 if i == self.frame_num:
#                     break
#                 filename = self.folder_list_clean[m] + f
#                 image = cv2.imread(filename)
#                 # image = image[:256, :320, :]
#                 if images is None:
#                     images = image
#                 else:
#                     images = np.concatenate((images, image), axis=2)
#             #     filename = self.folder_list_clean[m] + f
#             #     image = cv2.imread(filename)
#             #     images_list = images_list + (image,)
#             # images = np.concatenate(images_list, axis=2)
#         elif mode == 'LR':
#             images_list = tuple()
#             for i, f in enumerate(self.file_list):
#                 # print(i)
#                 # print(self.folder_list_corrupted[m])
#                 # print(f)
#                 if i == self.frame_num:
#                     break
#                 filename = self.folder_list_corrupted[m] + f
#                 image = cv2.imread(filename)
#                 # image = image[:256, :512, :]
#                 if images is None:
#                     images = image
#                 else:
#                     images = np.concatenate((images, image), axis=2)
#             #     filename = self.folder_list_corrupted[m] + f
#             #     image = cv2.imread(filename)
#             #     images_list = images_list + (image,)
#             # images = np.concatenate(images_list, axis=2)
#         elif mode == 'SR':
#             images_list = tuple()
#             for i, f in enumerate(self.file_list):
#                 if i == self.frame_num:
#                     break
#                 filename = self.folder_list_MDSR[m] + f
#                 image = cv2.imread(filename)
#                 # image = image[:256, :512, :]
#                 if images is None:
#                     images = image
#                 else:
#                     images = np.concatenate((images, image), axis=2)
#             #     filename = self.folder_list_MDSR[m] + f
#             #     image = cv2.imread(filename)
#             #     images = np.concatenate(images_list, axis=2)
#             # images = np.concatenate(images_list, axis=2)
#         print(images.shape)
#         return images
#
#     def __len__(self):
#         return self.N
#
#     def __getitem__(self, idx):
#         if self.is_train:
#             m = idx // self.frame_num
#             f_st = 0
#         else:
#             m = idx
#             f_st = 0
#         # self.window_size = self.frame_num_list[idx]
#
#         sample = dict()
#
#         sample['input_img1_LR'] = self.get_frame(m, f_st, mode='LR')
#         if self.transform:
#             sample['input_img1_LR'] = self.transform(sample['input_img1_LR'])
#
#         sample['input_img1_HR'] = self.get_frame(m, f_st, mode='HR')
#         if self.transform:
#             sample['input_img1_HR'] = self.transform(sample['input_img1_HR'])
#
#         sample['input_img1_SR'] = self.get_frame(m, f_st, mode='SR')
#         if self.transform:
#             sample['input_img1_SR'] = self.transform(sample['input_img1_SR'])
#
#         sample['input_img2_HR'] = self.get_frame(m, self.frame_num-1, mode='HR')
#         if self.transform:
#             sample['input_img2_HR'] = self.transform(sample['input_img2_HR'])
#         sample['input_LR'] = self.get_all_frames(m, mode='LR')
#         if self.transform:
#             sample['input_LR'] = self.transform(sample['input_LR'])
#         sample['input_HR'] = self.get_all_frames(m, mode='HR')
#         if self.transform:
#             sample['input_HR'] = self.transform(sample['input_HR'])
#
#         if self.require_seqid:
#             seq_id = self.folder_list_clean[m]
#             return sample, seq_id
#         else:
#             return sample
#
# if __name__ == "__main__":
#     data_path_corrupted = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_noise/'
#     data_path_MDSR = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_upsampled_MDSR/'
#     data_path_clean = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences/'
#     data_list_file = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sep_trainlist.txt'
#     # composed = transforms.Compose([transforms.RandomCrop((128,128)),
#     #                                 transforms.ToTensor()])
#
#     composed = transforms.Compose([transforms.ToTensor()])
#     dataset = VimeoDataset(data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, frame_window_size=2,
#                            transform=composed)
#
#     #### test pytorch dataset
#     # print(len(dataset))
#
#     # fig = plt.figure()
#     # plt.axis('off')
#     # plt.ioff()
#     # im = plt.imshow(np.zeros((dataset.H, dataset.W, 3)), vmin=0, vmax=1)
#
#     # for i in range(len(dataset)-1, 0, -1):
#     #     sample = dataset[i]
#     #     for t in sample:
#     #         print(t, sample[t].size())
#     #         im.set_data(sample[t].numpy().transpose(1,2,0))
#     #         plt.pause(0.1)
#     # exit()
#
#     #### test dataloader
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)
#     print(len(dataset), len(dataloader))
#
#     import cPickle as pickle
#     import os
#
#     img_name = ['img1_LR', 'img1_SR', 'img1_HR', 'img2_HR']
#     for i_batch, sample_batched in enumerate(dataloader):
#         # print(i_batch, sample_batched['gt'].size())
#         # visualization
#         images_batch = sample_batched['input_img1_LR']
#         batch_size = images_batch.size()[0]
#         im_size = images_batch.size()[1:]
#
#         print(i_batch)
#         save_dir = './vimeo_sr/'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         f = open(save_dir + str(i_batch + 1), 'wb')
#         pickle.dump(sample_batched, f)
#         f.close()
#
#         # print(batch_size, im_size)
#         # grid = utils.make_grid(images_batch, nrow=2)
#         # plt.imshow(grid.numpy().transpose(1,2,0))
#         # plt.show()
#
#         # observe 4th batch and stop.
#         # if i_batch == 3:
#         #     plt.figure()
#         #     show_landmarks_batch(sample_batched)
#         #     plt.axis('off')
#         #     plt.ioff()
#         #     plt.show()
#         #     break
