import sys
import os
from optparse import OptionParser
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import time
import cPickle as pickle
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2

sys.path.insert(0, './ref_utils/')
sys.path.insert(0, './Model/')
from Model import Crossnetpp_Multiflow, Crossnetpp_Multiflow2, Crossnetpp_MultiflowFusion, Crossnetpp_MultiflowFusion2, \
    Crossnetpp_MultiflowFusion3, Crossnetpp_MultiflowFusion4, Crossnetpp_MultiflowFusion5
from Model import Discriminator

from VimeoDataset import VimeoDataset
from MpiiDataset import MpiiDataset
import matplotlib.pyplot as plt
import CustomLoss
from sift_extractor import SiftExtractor
from skimage.measure import compare_ssim

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def gen_flow_label(sift_extractor, buff, flow):
#     input_img1_HR = np.array(buff['input_img1_HR'] * 255, dtype=np.uint8)
#     input_img2_HR = np.array(buff['input_img2_HR'] * 255, dtype=np.uint8)
#
#     flow_label = flow.detach().cpu().numpy().copy()
#     B, C, H, W = input_img1_HR.shape
#     for b in range(B):
#
#         img1 = input_img1_HR[b].transpose(1, 2, 0)
#         img2 = input_img2_HR[b].transpose(1, 2, 0)
#         lm1, lm2 = sift_extractor.get_matched_landmark(img1, img2)
#         if lm1 is None and lm2 is None:
#             continue
#
#         disparity = lm2 - lm1
#         for idx in range(disparity.shape[0]):
#             # print (flow_label[b,:,lm1[idx,1],lm1[idx,0]],disparity[idx,:])
#             flow_label[b, :, lm1[idx, 1], lm1[idx, 0]] = disparity[idx, :]
#
#     return torch.from_numpy(flow_label)


def save_img(buff, warp_img2_HR, fine_img1_SR, warp_HR, file_dir):
    # print(file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    keys = buff.keys()
    for key in keys:
        if key == 'input_LR' or key == 'input_HR':
            for i in range(frame_num):
                temp = buff[key][:, i * 3: (i+1) * 3, :, :]
                img = Image.fromarray(
                    np.array(temp.numpy()[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
                img.save(file_dir + key + '_%04d.png' % i)
            continue
        img = Image.fromarray(np.array(buff[key].numpy()[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        img.save(file_dir + key + '.png')
    for i in range(frame_num):
        temp = warp_HR[:, i * 3: (i + 1) * 3, :, :]
        # print(temp.shape)
        img = Image.fromarray(
            np.array(temp.cpu().numpy()[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        img.save(file_dir + 'warp_HR_%04d.png' % i)

    warp_img2_HR = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
    img = Image.fromarray(np.array(warp_img2_HR[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
    img.save(file_dir + 'warp_img2_HR.png')
    sr_img = np.clip(fine_img1_SR.cpu().numpy(), 0.0, 1.0)
    img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
    img.save(file_dir + 'sr.png')


def eval(net, testloader, len_testset, config, iter_count = 0):
    net.eval()
    sum_psnr = 0.0
    sum_ssim = 0.0
    time_start = time.time()
    print('---------start eval--------')
    print(len_testset, len(testloader))
    for iter_, data in enumerate(testloader):

        # if iter_ >= 10:
        #    continue
        buff, seqid = data
        label_img = buff['input_img1_HR'].numpy()
        with torch.no_grad():
            # if config['net_type'] == 'multiflow' or config['net_type'] == 'multiflowfusion'
            warp_img2_HR, fine_img1_SR, warp_HR = net(buff, vimeo=True, require_flow=False)
            # warp_img2_HR, fine_img1_SR = net(buff, vimeo=True, require_flow=False)
            save_img(buff, warp_img2_HR, fine_img1_SR, warp_HR, './%s/%04d/%s/' % (config['img_save_path'], iter_count, seqid[0].strip().split('/')[-2]))
            fine_img1_SR = fine_img1_SR.cpu().numpy()
            for i in range(label_img.shape[0]):
                sum_psnr += psnr(fine_img1_SR[i], label_img[i])
                ssim_ = compare_ssim(cv2.cvtColor(np.array(fine_img1_SR[i] * 255.0, dtype=np.uint8).transpose(1, 2, 0),
                                                  cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(np.array(label_img[i] * 255.0, dtype=np.uint8).transpose(1, 2, 0),
                                                  cv2.COLOR_BGR2GRAY))
                sum_ssim += ssim_
                time_cost = time.time() - time_start
    res_psnr = sum_psnr / len_testset
    res_ssim = sum_ssim / len_testset

    time_cost = time.time() - time_start
    if res_psnr > config['best_eval']:
        config['best_eval'] = res_psnr
    print('------PSNR: %.2f, SSIM: %.2f, so far the best is: %.2f, time: %.2f--------' % (
    res_psnr, res_ssim, config['best_eval'], time_cost))


def train_net(net, gpu=False, config={}):
    dataset_train = config['dataset_train']
    dataset_test = config['dataset_test']

    discriminator = config['discriminator']

    len_testset = len(dataset_test)

    trainloader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=6)
    testloader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False, num_workers=6)

    print('trainset: %d, trainloder:%d ' % (len(dataset_train), len(trainloader)))
    print('testset: %d, testloder:%d ' % (len(dataset_test), len(testloader)))
    print('Starting training...')

    if config['optim'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00005)

    if discriminator is not None:
        criterionBCE = nn.BCEWithLogitsLoss(size_average=True)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], weight_decay=0.00005)

    if config['loss'] == 'EuclideanLoss':
        criterion = CustomLoss.EuclideanLoss()
    elif config['loss'] == 'CharbonnierLoss':
        criterion = CustomLoss.CharbonnierLoss()
    elif config['loss'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        print
        'None loss type'
        sys.exit(0)

    sift_extractor = config['sift_extractor']
    criterion_warp = CustomLoss.EuclideanLoss()
    loss_count = np.zeros(3, dtype=np.float32)
    time_start = time.time()

    iter_count = config['checkpoint']

    net.train()
    for epoch in range(config['epoch'], 5000):
        for iter_, buff in enumerate(trainloader):

            label_img = buff['input_img1_HR']
            input_HR = buff['input_HR'].cuda()

            if gpu:
                label_img = label_img.cuda()
            # print (label_img.size())

            warp_img2_HR, fine_img1_SR, warp_HR = net(buff, vimeo=True, require_flow=False)

            # flow_label = gen_flow_label(sift_extractor, buff, flow_s1_12_1)
            # if gpu:
            #     flow_label = flow_label.cuda()

            loss_1 = criterion_warp(warp_img2_HR, label_img)
            # loss_2 = criterion_warp(flow_s1_12_1, flow_label)
            loss_3 = criterion(fine_img1_SR, label_img)
            loss_4 = criterion_warp(warp_HR, input_HR) / (input_HR.shape[1] // 3)

            loss_count[0] += config['w1'] * loss_1.item()
            # loss_count[1] += loss_2.item()
            loss_count[1] += config['w2'] * loss_3.item()
            loss_count[2] += config['w3'] * loss_4.item()

            loss_d_display = 0.0
            loss_g_display = 0.0
            # GAN loss
            if discriminator is not None and iter_count % 2 == 0:

                prediction_fake = discriminator(fine_img1_SR.detach())
                prediction_real = discriminator(label_img)

                logits0 = Variable(torch.ones(prediction_fake.size()).cuda(), requires_grad=False)
                logits1 = Variable(torch.zeros(prediction_fake.size()).cuda(), requires_grad=False)

                # Fake samples
                loss_d_fake = criterionBCE(prediction_fake, logits0)
                # Real samples
                loss_d_real = criterionBCE(prediction_real, logits1)

                # Combined
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d_display = loss_d.item()
                # Backprop and update
                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()

            else:

                prediction_fake = discriminator(fine_img1_SR)
                logits1 = Variable(torch.zeros(prediction_fake.size()).cuda(), requires_grad=False)

                loss_g = criterionBCE(prediction_fake, logits1)
                loss_g_display = loss_g.item()
                w_gan = 0.0001
                # print
                loss = config['w1'] * loss_1 + config['w2'] * loss_3 + config['w3'] * loss_4 + w_gan * loss_g
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (iter_count + 1) % config['snapshot'] == 0:

                if not os.path.exists(config['checkpoints_dir']):
                    os.makedirs(config['checkpoints_dir'])
                torch.save(net.state_dict(),
                           config['checkpoints_dir'] + 'CP{}.pth'.format(iter_count + 1))
                if discriminator is not None:
                    torch.save(discriminator.state_dict(),
                               config['checkpoints_dir'] + 'D_CP{}.pth'.format(iter_count + 1))

                print('Checkpoint {} saved !'.format(iter_count + 1))
                eval(net, testloader, len_testset, config, (iter_count + 1) // config['snapshot'])

            if (iter_count + 1) % config['display'] == 0:
                time_end = time.time()
                time_cost = time_end - time_start

                # ------------------------------------------------
                pre_npy_2 = fine_img1_SR.data.cpu().numpy()
                label_img_npy = label_img.data.cpu().numpy()

                psnr_2 = 0
                for i in range(pre_npy_2.shape[0]):
                    # psnr_1 += psnr(pre_npy_1[i],label_img_npy[i]) / pre_npy_1.shape[0]
                    psnr_2 += psnr(pre_npy_2[i], label_img_npy[i]) / pre_npy_2.shape[0]

                loss_count = loss_count / config['display']
                print(
                    'iter:%d    time: %.2fs / %diters  lr: %.8f  LossR: %.3f %.3f %.3f LossGAN: %.3f %.3f psnr: %.2f' % (
                    iter_count + 1, time_cost, config['display'], config['lr'], loss_count[0], loss_count[1],
                    loss_count[2], loss_d, loss_g, psnr_2))

                loss_count[:] = 0
                time_start = time.time()

            if (iter_count + 1) % config['step_size'] == 0:
                config['lr'] = config['lr'] * config['gamma']
                if config['optim'] == 'SGD':
                    optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9,
                                          weight_decay=0.0005)
                elif config['optim'] == 'Adam':
                    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00005)

                    if discriminator is not None:
                        optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], weight_decay=0.00005)

            iter_count += 1


def get_args():
    parser = OptionParser()

    parser.add_option('--batch_size', dest='batch_size', default=4,
                      type='int', help='batch size')
    parser.add_option('--lr', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--checkpoint_file', dest='load',
                      default=False, help='load file model')

    parser.add_option('--discriminator_file', dest='discriminator_file',
                      default=None, help='load discriminator checkpoint file')

    parser.add_option('--checkpoint', dest='checkpoint', default=0, type='int', help='snapshot')

    parser.add_option('--epoch', dest='epoch', default=0, type='int', help='Interrupted epoch')

    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=8, help='downscaling factor of LR')

    parser.add_option('--loss', dest='loss', default='EuclideanLoss', help='loss type')

    parser.add_option('--dataset', dest='dataset', default='Vimeo', help='dataset type')

    parser.add_option('--gamma', dest='gamma', type='float', default=0.2, help='lr decay')

    parser.add_option('--step_size', dest='step_size', type='float', default=60000, help='step_size')

    parser.add_option('--max_iter', dest='max_iter', default=1000000, type='int', help='max_iter')

    parser.add_option('--checkpoints_dir', dest='checkpoints_dir', default='./checkpoints/', help='checkpoints_dir')

    parser.add_option('--snapshot', dest='snapshot', default=5000, type='float', help='snapshot')

    parser.add_option('--display', dest='display', default=10, type='float', help='display')

    parser.add_option('--optim', dest='optim', default='SGD', help='optimizer type')

    parser.add_option('--pretrained', dest='pretrained', default=None, help='optimizer type')
    parser.add_option('--mode', dest='mode', default='train', help='test_file')
    parser.add_option('--test_file', dest='test_file', default='sep_testlist_small.txt', help='train or test')
    parser.add_option('--w1', dest='w1', default=1.0, type='float', help='coarse weight')
    parser.add_option('--w2', dest='w2', default=1.0, type='float', help='fine weight')
    parser.add_option('--w3', dest='w3', default=1.0, type='float', help='multi-frame flow weight')
    parser.add_option('--gpu_id', dest='gpu_id', default=0, type='int', help='gpu_id')
    parser.add_option('--frame_num', dest='frame_num', default=2, type='int', help='frames number')
    parser.add_option('--with_GAN_loss', dest='with_GAN_loss', default=0, type='int', help='use GAN loss')

    parser.add_option('--img_save_path', dest='img_save_path', default=None, help='save path for evaluation img')
    parser.add_option('--net_type', dest='net_type', default=None, help='choose the network model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    args = get_args()

    if args.gpu_id != 0:
        print('gpuid', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)

    net_type = args.net_type
    if net_type == 'original':
        net = Crossnetpp_Original()
    elif net_type == 'multiflow':
        net = Crossnetpp_Multiflow()
    elif net_type == 'multiflow2':
        net = Crossnetpp_Multiflow2()
    elif net_type == 'multiflowfusion':
        net = Crossnetpp_MultiflowFusion()
    elif net_type == 'multiflowfusion2':
        net = Crossnetpp_MultiflowFusion2()
    elif net_type == 'multiflowfusion3':
        net = Crossnetpp_MultiflowFusion3()
    elif net_type == 'multiflowfusion4':
        net = Crossnetpp_MultiflowFusion4()
    elif net_type == 'multiflowfusion5':
        net = Crossnetpp_MultiflowFusion5()

    dataset_name = args.dataset
    scale = args.scale
    frame_num = args.frame_num

    if dataset_name == 'Vimeo':
        data_path_corrupted = '../../../../fileserver/haitian/Warp_layer/vimeo_septuplet/sequences_blur/'
        data_path_MDSR = '../../../../fileserver/haitian/Warp_layer/vimeo_septuplet/sequences_upsampled_MDSR/'
        data_path_clean = '../../../../fileserver/haitian/Warp_layer/vimeo_septuplet/sequences/'
        train_list_file = '../../../../fileserver/haitian/Warp_layer/vimeo_septuplet/sep_trainlist.txt'
        test_list_file = '../../../../fileserver/haitian/Warp_layer/vimeo_septuplet/' + args.test_file
        # composed = transforms.Compose([transforms.RandomCrop((128,128)),transforms.ToTensor()])
        composed = transforms.Compose([transforms.ToTensor()])
        dataset_train = VimeoDataset(data_path_corrupted, data_path_clean, data_path_MDSR, train_list_file,
                                     frame_num=frame_num, transform=composed)
        dataset_test = VimeoDataset(data_path_corrupted, data_path_clean, data_path_MDSR, test_list_file,
                                    frame_num=frame_num, transform=composed, is_train=False,
                                    require_seqid=True)

    if dataset_name == 'MPII':
        data_path_corrupted = '../../../../fileserver/yaping/data/crossnet/MPII_1_640_448/LR_4x/'
        data_path_MDSR = '../../../../fileserver/yaping/data/crossnet/MPII_1_640_448/LR_4x/'
        data_path_clean = '../../../../fileserver/yaping/data/crossnet/MPII_1_640_448/HR/'
        train_list_file = '../../../../fileserver/yaping/data/crossnet/MPII_1_640_448/MPII_1_640_448.txt'
        test_list_file = '../../../../fileserver/yaping/data/crossnet/MPII_1_640_448/MPII_1_640_448.txt'
        # composed = transforms.Compose([transforms.RandomCrop((128,128)),transforms.ToTensor()])
        composed = transforms.Compose([transforms.ToTensor()])
        dataset_train = MpiiDataset(data_path_corrupted, data_path_clean, data_path_MDSR, train_list_file,
                                    frame_num=frame_num, transform=composed)

        test_path_corrupted = '../../../../fileserver/yaping/data/crossnet/MPII_2_640_448/LR_4x/'
        test_path_MDSR = '../../../../fileserver/yaping/data/crossnet/MPII_2_640_448/LR_4x/'
        test_path_clean = '../../../../fileserver/yaping/data/crossnet/MPII_2_640_448/HR/'
        test_list_file = '../../../../fileserver/yaping/data/crossnet/MPII_2_640_448/MPII_2_640_448.txt'
        dataset_test = MpiiDataset(test_path_corrupted, test_path_clean, test_path_MDSR, test_list_file,
                                   frame_num=frame_num, transform=composed, is_train=True, require_seqid=True)

    if dataset_name == 'DAVIS':
        dataset_train = None
        test_path_corrupted = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/DAVIS_2017_and_2019_rigid/LR_4x/'
        test_path_MDSR = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/DAVIS_2017_and_2019_rigid/LR_4x/'
        test_path_clean = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/DAVIS_2017_and_2019_rigid/HR/'
        test_list_file = '/fileserver/tanyang/projects/ref_sr_ytan/dataset/MPII_from_yaping/DAVIS_2017_and_2019_rigid/DAVIS_rigid.txt'
        composed = transforms.Compose([transforms.ToTensor()])
        dataset_test = MpiiDataset(test_path_corrupted, test_path_clean, test_path_MDSR, test_list_file,
                                   min_window_size=3, transform=composed, is_train=False, require_seqid=True)

    config = {}
    config['dataset_train'] = dataset_train
    config['dataset_test'] = dataset_test
    config['snapshot'] = args.snapshot
    config['display'] = args.display
    config['lr'] = args.lr
    config['batch_size'] = args.batch_size
    config['step_size'] = args.step_size
    config['gamma'] = args.gamma
    config['checkpoints_dir'] = args.checkpoints_dir
    config['loss'] = args.loss
    config['checkpoint'] = args.checkpoint
    config['epoch'] = args.epoch
    config['optim'] = args.optim
    config['w1'] = args.w1
    config['w2'] = args.w2
    config['w3'] = args.w3
    config['best_eval'] = 0.0
    config['sift_extractor'] = SiftExtractor()
    config['discriminator'] = None
    config['img_save_path'] = args.img_save_path
    config['net_type'] = args.net_type

    if args.with_GAN_loss == 1:
        config['discriminator'] = Discriminator(input_size=(256, 448))

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

        if config['discriminator'] is not None and args.discriminator_file is not None:
            config['discriminator'].load_state_dict(args.discriminator_file)

    # if args.pretrained:
    #     MW_2stage_model = torch.load(args.load)
    #     print('Model loaded from {}'.format(args.load))
    #     cur_model = net.state_dict()
    #     # print MW_2stage_model.keys()
    #     FlowNet_s1 = {'FlowNet_s1.' + k[21::]: v for k, v in MW_2stage_model.items() if
    #                   'FlowNet_s1.' + k[21::] in cur_model and k[0:21] == 'MWNet_coarse.FlowNet.'}
    #     FlowNet_s2 = {'FlowNet_s2.' + k[19::]: v for k, v in MW_2stage_model.items() if
    #                   'FlowNet_s2.' + k[19::] in cur_model and k[0:19] == 'MWNet_fine.FlowNet.'}
    #
    #     encoder_decoder = {k[11::]: v for k, v in MW_2stage_model.items() if k[11::] in cur_model}
    #     # print FlowNet_s1.keys(),FlowNet_s2.keys()
    #     # print
    #     # encoder_decoder
    #     cur_model.update(encoder_decoder)
    #     cur_model.update(FlowNet_s1)
    #     cur_model.update(FlowNet_s2)
    #     net.load_state_dict(cur_model)
    #     print 'finetuning...'

    if args.gpu:
        net.cuda()
        if config['discriminator'] is not None:
            config['discriminator'].cuda()

    if args.mode == 'train':
        try:
            train_net(net=net, gpu=args.gpu, config=config)

        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    elif args.mode == 'test':
        len_testset = len(dataset_test)
        testloader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False, num_workers=6)
        eval(net, testloader, len_testset, config)
