from Dataset import Dataset
import h5py
import numpy as np
import random
class GigaDataset(Dataset):
    def __init__(self, filename=None, scale=4, MDSR_as_bilinear=False):
        assert scale==4
        super(GigaDataset, self).__init__(filename,scale,MDSR_as_bilinear)
    def genViewPosition(self,view_mode,block_idx=None):

        if (view_mode == 'Random'):
            rnd=self.rng_viewpoint.randint(0,2)
        elif (view_mode == 'left'):
            rnd=0
        elif (view_mode == 'right'):
            rnd=1
        elif view_mode == 'test':
            rnd = 0
        else:
            RuntimeError('viewmode not supported')
        return rnd
    def loadArrays(self, filename, scale=8, MDSR_as_bilinear=False):
        f = h5py.File(filename, 'r')
        self.scale = scale
        self.arrays['img_HR'] = f.get('/HR')
        self.arrays['img_LR'] = f.get('/LR')
        self.arrays['img_LR_upsample'] = f.get('/LR_upsample')
        self.arrays['img_MDSR'] = f.get('/SR')

        print('loading dataset: ', filename)
        print('img_HR ', self.arrays['img_HR'].shape)
        print('img_LR ', self.arrays['img_LR'].shape)
        print('img_LR_upsample ', self.arrays['img_LR_upsample'].shape)
        if not self.arrays['img_MDSR'] is None:
            print('img_MDSR ', self.arrays['img_MDSR'].shape)
        else:
            print('cannot find img_MDSR')
        self.size_N = self.arrays['img_HR'].shape[0]
        self.size_C = self.arrays['img_HR'].shape[3]
        self.size_H = self.arrays['img_HR'].shape[4]
        self.size_W = self.arrays['img_HR'].shape[5]
    def nextBatch_new(self, batchsize=8, shuffle=False, view_mode='Random', specified_view=None, augmentation=False,
                      offset_augmentation=False, index_inc=True, crop_shape=None, random_crop=True, SR=True,
                      Dual=False):

        buff = dict()
        assert Dual is False
        input_img1_LR = np.zeros([batchsize, 3, self.size_H, self.size_W], dtype=np.float32)
        input_img1_HR = np.zeros([batchsize, 3, self.size_H, self.size_W], dtype=np.float32)
        input_img2_HR = np.zeros([batchsize, 3, self.size_H, self.size_W], dtype=np.float32)
        if SR:
            input_img1_SR = np.zeros([batchsize, 3, self.size_H, self.size_W], dtype=np.float32)
        idx_list = self.genIndex_list(batchsize, shuffle, index_inc=index_inc)
        buff['idx_list'] = idx_list
        # viewpoint is used only for diag view case such as (1,1),(3,3)...
        buff['view_point'] = None
        x = self.genViewPosition(view_mode,block_idx=specified_view)
        x1,y1,x2,y2=x,0,1-x,0
        for k in range(batchsize):
            # generate img number
            idx_img = idx_list[k]
            # LR
            input_img1_LR[k, :, :, :] = np.asarray(self.arrays['img_LR_upsample'][idx_img, y1, x1, :, :, :],
                                                   dtype=np.float32) / 255.0
            input_img1_HR[k, :, :, :] = np.asarray(self.arrays['img_HR'][idx_img, y1, x1, :, :, :],
                                                   dtype=np.float32) / 255.0
            input_img2_HR[k, :, :, :] = np.asarray(self.arrays['img_HR'][idx_img, y2, x2, :, :, :],
                                                   dtype=np.float32) / 255.0
            # SR
            if SR:
                input_img1_SR[k, :, :, :] = np.asarray(self.arrays['img_MDSR'][idx_img, y1, x1, :, :, :],
                                                       dtype=np.float32) / 255.0

        # data augmentation
        if augmentation:
            augmentation_config = self.augmentation_array_config()
            input_img1_LR = self.augmentation_array(input_img1_LR, augmentation_config)
            input_img1_HR = self.augmentation_array(input_img1_HR, augmentation_config)
            input_img2_HR = self.augmentation_array(input_img2_HR, augmentation_config)
            if SR:
                input_img1_SR = self.augmentation_array(input_img1_SR, augmentation_config)

        if offset_augmentation:
            dx = self.rng_viewpoint_augmentation.randint(-14, 15)
            dy = self.rng_viewpoint_augmentation.randint(-14, 15)
            # print dx, dy
            input_img2_HR[:, :, max(0, 0 + dy):min(self.size_H, self.size_H + dy),
            max(0, 0 + dx):min(self.size_W, self.size_W + dx)] = input_img2_HR[:, :,
                                                                 0:min(self.size_H, self.size_H + dy) - max(0, 0 + dy),
                                                                 0:min(self.size_W, self.size_W + dx) - max(0, 0 + dx)]

        # crop image
        if not crop_shape is None:

            off_x = 0
            off_y = 0
            if random_crop:
                off_x_range=self.size_W-crop_shape[1]
                off_y_range=self.size_H-crop_shape[0]
                off_x = random.randint(0, off_x_range)
                off_y = random.randint(0, off_y_range)
            input_img1_LR = input_img1_LR[:, :, off_y:off_y + crop_shape[0], off_x:off_x + crop_shape[1]]
            input_img1_HR = input_img1_HR[:, :, off_y:off_y + crop_shape[0], off_x:off_x + crop_shape[1]]
            input_img2_HR = input_img2_HR[:, :, off_y:off_y + crop_shape[0], off_x:off_x + crop_shape[1]]
            if SR:
                input_img1_SR = input_img1_SR[:, :, off_y:off_y + crop_shape[0], off_x:off_x + crop_shape[1]]

        # pack buffer
        buff['input_img1_LR'] = input_img1_LR
        buff['input_img1_HR'] = input_img1_HR
        buff['input_img2_HR'] = input_img2_HR
        if SR:
            buff['input_img1_SR'] = input_img1_SR
        return buff
if __name__ == '__main__':
    test_dataset=GigaDataset(r'C:\Users\haffm\Documents\GitHub\cross_net_pytorch\external_packages\make_giga_datset\test.h5')
    x=test_dataset.nextBatch_new(batchsize=2, shuffle=True,
                                           augmentation=True,
                                           crop_shape=(320,512))
    print('done')
