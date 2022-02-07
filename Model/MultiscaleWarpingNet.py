import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from FlowNet_model import FlowNet
from FlowNet_model import FlowNet_dilation
from Backward_warp_layer import Backward_warp
from SupervisedFlowNet import SupervisedFlowNet
import numpy as np
from PIL import Image

class MultiscaleWarpingNet(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()


    def forward(self, buff,mode = 'input_img2_HR',flow_visible=False):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
        if mode == 'input_img2_LR':
            input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
            flow = self.FlowNet(input_img1_LR, input_img2_LR)
        elif mode == 'input_img2_HR':
            flow = self.FlowNet(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_HR':
            flow = self.FlowNet(input_img1_HR, input_img2_HR) 
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)        

        final_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        if flow_visible:
            return final_output,flow
        else:
            return final_output



#supervised by HR&HR flow
class MultiscaleWarpingNet2(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet2, self).__init__()
        self.SupervisedFlowNet = SupervisedFlowNet()
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()


    def forward(self, buff):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        flow_coarse,flow_fine = self.SupervisedFlowNet(buff)

        flow_12_1 = flow_coarse['flow_12_1']
        flow_12_2 = flow_coarse['flow_12_2']
        flow_12_3 = flow_coarse['flow_12_3']
        flow_12_4 = flow_coarse['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)        

        final_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return final_output,flow_coarse,flow_fine


#encoder independent
class MultiscaleWarpingNet3(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet3, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder1 = Encoder(3)
        self.Encoder2 = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()


    def forward(self, buff,mode = 'input_img2_HR',vimeo=False):

        if vimeo:
            input_img1_LR = buff['input_img1_LR'].cuda()
            input_img1_HR = buff['input_img1_HR'].cuda()
            input_img1_SR = buff['input_img1_SR'].cuda()
            input_img2_HR = buff['input_img2_HR'].cuda()
        else:
            input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
            input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
            input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
            input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        # input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
        if mode == 'input_img2_LR':
            input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
            flow = self.FlowNet(input_img1_LR, input_img2_LR)
        elif mode == 'input_img2_HR':
            flow = self.FlowNet(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_HR':
            flow = self.FlowNet(input_img1_HR, input_img2_HR)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder1(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder2(input_img2_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)

        final_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return final_output


#only encoding HR 
class MultiscaleWarpingNet4(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet4, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_3 = UNet_decoder_3()


    def forward(self, buff,mode = 'input_img2_HR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
        if mode == 'input_img2_LR':
            input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
            flow = self.FlowNet(input_img1_LR, input_img2_LR)
        elif mode == 'input_img2_HR':
            flow = self.FlowNet(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_HR':
            flow = self.FlowNet(input_img1_HR, input_img2_HR)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        #SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)

        final_output = self.UNet_decoder_3(warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return final_output


#the input of Encoder is concatenation of LR & HR
class MultiscaleWarpingNet5(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet5, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(6)
        self.UNet_decoder_3 = UNet_decoder_3()

    def forward(self, buff,mode = 'input_img2_HR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
        if mode == 'input_img2_LR':
            input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
            flow = self.FlowNet(input_img1_LR, input_img2_LR)
        elif mode == 'input_img2_HR':
            flow = self.FlowNet(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_HR':
            flow = self.FlowNet(input_img1_HR, input_img2_HR)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']


        concat0 = torch.cat((input_img1_SR,input_img2_HR),1)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(concat0)
        
        
        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)

        final_output = self.UNet_decoder_3(warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return final_output


#shared crossnet for two-stage 
# class MultiscaleWarpingNet6(nn.Module):
#     def __init__(self):
#         super(MultiscaleWarpingNet6, self).__init__()
#         self.FlowNet = FlowNet(6)
#         self.Backward_warp = Backward_warp()
#         self.Encoder = Encoder(3)
#         self.UNet_decoder_2 = UNet_decoder_2()
#
#     def forward(self, buff,mode = 'input_img2_HR'):
#
#         input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
#         input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
#         input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
#
#         # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
#         input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
#         # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
#         if mode == 'input_img2_LR':
#             input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
#             flow_s1 = self.FlowNet(input_img1_LR, input_img2_LR)
#         elif mode == 'input_img2_HR':
#             flow_s1 = self.FlowNet(input_img1_LR, input_img2_HR)
#         elif mode == 'input_img1_HR':
#             flow_s1 = self.FlowNet(input_img1_HR, input_img2_HR)
#
#         SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
#         HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)
#
#         flow_s1_12_1 = flow_s1['flow_12_1']
#         flow_s1_12_2 = flow_s1['flow_12_2']
#         flow_s1_12_3 = flow_s1['flow_12_3']
#         flow_s1_12_4 = flow_s1['flow_12_4']
#
#         warp_s1_21_conv1 = self.Backward_warp(HR2_conv1, flow_s1_12_1)
#         warp_s1_21_conv2 = self.Backward_warp(HR2_conv2, flow_s1_12_2)
#         warp_s1_21_conv3 = self.Backward_warp(HR2_conv3, flow_s1_12_3)
#         warp_s1_21_conv4 = self.Backward_warp(HR2_conv4, flow_s1_12_4)
#
#         refSR_1 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s1_21_conv1,warp_s1_21_conv2, warp_s1_21_conv3,warp_s1_21_conv4)
#
#         flow_s2 = self.FlowNet(refSR_1, input_img2_HR)
#         flow_s2_12_1 = flow_s2['flow_12_1']
#         flow_s2_12_2 = flow_s2['flow_12_2']
#         flow_s2_12_3 = flow_s2['flow_12_3']
#         flow_s2_12_4 = flow_s2['flow_12_4']
#
#         warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
#         warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
#         warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
#         warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
# 	refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,warp_s2_21_conv2, warp_s2_21_conv3,warp_s2_21_conv4)
#
#
#
#         return refSR_1,refSR_2

#shared encoder+decoder but independent flownet
class MultiscaleWarpingNet7(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet7, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff,mode = 'input_img2_HR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()
        if mode == 'input_img2_LR':
            input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
            flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_LR)
        elif mode == 'input_img2_HR':
            flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_HR':
            flow_s1 = self.FlowNet_s1(input_img1_HR, input_img2_HR)

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        flow_s1_12_1 = flow_s1['flow_12_1']
        flow_s1_12_2 = flow_s1['flow_12_2']
        flow_s1_12_3 = flow_s1['flow_12_3']
        flow_s1_12_4 = flow_s1['flow_12_4']

        warp_s1_21_conv1 = self.Backward_warp(HR2_conv1, flow_s1_12_1)
        warp_s1_21_conv2 = self.Backward_warp(HR2_conv2, flow_s1_12_2)
        warp_s1_21_conv3 = self.Backward_warp(HR2_conv3, flow_s1_12_3)
        warp_s1_21_conv4 = self.Backward_warp(HR2_conv4, flow_s1_12_4)

        refSR_1 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s1_21_conv1,warp_s1_21_conv2, warp_s1_21_conv3,warp_s1_21_conv4)

        flow_s2 = self.FlowNet_s2(refSR_1, input_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,warp_s2_21_conv2, warp_s2_21_conv3,warp_s2_21_conv4)



        return refSR_1,refSR_2

#1st stage warped on input img, 2nd stage is crossnet
class MultiscaleWarpingNet8(nn.Module):

    def __init__(self, flownet_type = 'FlowNet_ori'):
        super(MultiscaleWarpingNet8, self).__init__()
        
        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff,vimeo=False, require_flow=False, encoder_input = 'input_img1_SR',flow_visible=False ):

        if vimeo:
            input_img1_LR = buff['input_img1_LR'].cuda()
            input_img1_HR = buff['input_img1_HR'].cuda()
            input_img1_SR = buff['input_img1_SR'].cuda()
            input_img2_HR = buff['input_img2_HR'].cuda()
        else:
            input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
            #input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
            input_img1_SR = torch.from_numpy(buff[encoder_input]).cuda()
            input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        print ('test>>>>>>>>>')
        print (input_img1_LR.shape, input_img2_HR.shape)
        flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        flow_s1_12_1 = flow_s1['flow_12_1']

        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,warp_s2_21_conv2, warp_s2_21_conv3,warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR,refSR_2,flow_s1,flow_s2
        else:
            if require_flow:
                return warp_img2_HR,refSR_2,flow_s1_12_1
            else:
                return warp_img2_HR,refSR_2


#1st stage warped on input img, 2nd stage is crossnet
class MultiscaleWarpingNet8_Denoise(nn.Module):

    def __init__(self,gray=False):
        super(MultiscaleWarpingNet8_Denoise, self).__init__()

        if gray:
            channel = 2
        else:
            channel = 6
        self.FlowNet_s1 = FlowNet(channel)
        self.FlowNet_s2 = FlowNet(channel)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(channel/2)
        self.UNet_decoder_4 = UNet_decoder_4()

    def forward(self, buff,vimeo=True,is_train=True ):

        if vimeo:
            input_img1_noise = buff['0_noise'].cuda()
            input_img2_noise = buff['1_noise'].cuda()
            input_img3_noise = buff['2_noise'].cuda()
            input_img1_clean = buff['0_clean'].cuda()
            input_img2_clean = buff['1_clean'].cuda()
            input_img3_clean = buff['2_clean'].cuda()


        #------1st stage: img1 to img2----------------------
        flow_s1_f12 = self.FlowNet_s1(input_img2_noise, input_img1_noise)
        flow_s1_f12_1 = flow_s1_f12['flow_12_1']
        warp_img1_noise = self.Backward_warp(input_img1_noise, flow_s1_f12_1)
        if is_train:
            warp_img1_clean = self.Backward_warp(input_img1_clean, flow_s1_f12_1)

        #------1st stage: img3 to img2----------------------
        flow_s1_f32 = self.FlowNet_s1(input_img2_noise, input_img3_noise)
        flow_s1_f32_1 = flow_s1_f32['flow_12_1']
        warp_img3_noise = self.Backward_warp(input_img3_noise, flow_s1_f32_1)
        if is_train:
            warp_img3_clean = self.Backward_warp(input_img3_clean, flow_s1_f32_1)


        img2_conv1, img2_conv2, img2_conv3, img2_conv4 = self.Encoder(input_img2_noise)
        #-----2nd stage: img1 to img2----------------------
        flow_s2_f12 = self.FlowNet_s2(input_img2_noise, warp_img1_noise)
        flow_s2_f12_1 = flow_s2_f12['flow_12_1']
        flow_s2_f12_2 = flow_s2_f12['flow_12_2']
        flow_s2_f12_3 = flow_s2_f12['flow_12_3']
        flow_s2_f12_4 = flow_s2_f12['flow_12_4']
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(warp_img1_noise)
        warp_s2_12_conv1 = self.Backward_warp(img1_conv1, flow_s2_f12_1)
        warp_s2_12_conv2 = self.Backward_warp(img1_conv2, flow_s2_f12_2)
        warp_s2_12_conv3 = self.Backward_warp(img1_conv3, flow_s2_f12_3)
        warp_s2_12_conv4 = self.Backward_warp(img1_conv4, flow_s2_f12_4)

        #-----2nd stage: img3 to img2----------------------
        flow_s2_f32 = self.FlowNet_s2(input_img2_noise, warp_img3_noise)
        flow_s2_f32_1 = flow_s2_f32['flow_12_1']
        flow_s2_f32_2 = flow_s2_f32['flow_12_2']
        flow_s2_f32_3 = flow_s2_f32['flow_12_3']
        flow_s2_f32_4 = flow_s2_f32['flow_12_4']
        img3_conv1, img3_conv2, img3_conv3, img3_conv4 = self.Encoder(warp_img3_noise)
        warp_s2_32_conv1 = self.Backward_warp(img3_conv1, flow_s2_f32_1)
        warp_s2_32_conv2 = self.Backward_warp(img3_conv2, flow_s2_f32_2)
        warp_s2_32_conv3 = self.Backward_warp(img3_conv3, flow_s2_f32_3)
        warp_s2_32_conv4 = self.Backward_warp(img3_conv4, flow_s2_f32_4)

        refSR_2 = self.UNet_decoder_4(img2_conv1, img2_conv2, img2_conv3, img2_conv4, warp_s2_12_conv1,warp_s2_12_conv2, warp_s2_12_conv3,warp_s2_12_conv4,warp_s2_32_conv1,warp_s2_32_conv2, warp_s2_32_conv3,warp_s2_32_conv4)
        if is_train:
            return warp_img1_clean,warp_img3_clean,flow_s1_f12_1,flow_s1_f32_1,refSR_2
        else:
            return refSR_2


#crossnet for denoise
class MultiscaleWarpingNet_Denoise(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet_Denoise, self).__init__()
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_4 = UNet_decoder_4()

    def forward(self, buff,vimeo=True ):

        if vimeo:
            input_img1_noise = buff['0_noise'].cuda()
            input_img2_noise = buff['1_noise'].cuda()
            input_img3_noise = buff['2_noise'].cuda()


        img2_conv1, img2_conv2, img2_conv3, img2_conv4 = self.Encoder(input_img2_noise)
        flow_s2_f12 = self.FlowNet_s2(input_img2_noise, input_img1_noise)
        flow_s2_f12_1 = flow_s2_f12['flow_12_1']
        flow_s2_f12_2 = flow_s2_f12['flow_12_2']
        flow_s2_f12_3 = flow_s2_f12['flow_12_3']
        flow_s2_f12_4 = flow_s2_f12['flow_12_4']
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(input_img1_noise)
        warp_s2_12_conv1 = self.Backward_warp(img1_conv1, flow_s2_f12_1)
        warp_s2_12_conv2 = self.Backward_warp(img1_conv2, flow_s2_f12_2)
        warp_s2_12_conv3 = self.Backward_warp(img1_conv3, flow_s2_f12_3)
        warp_s2_12_conv4 = self.Backward_warp(img1_conv4, flow_s2_f12_4)

        #-----2nd stage: img3 to img2----------------------
        flow_s2_f32 = self.FlowNet_s2(input_img2_noise, input_img3_noise)
        flow_s2_f32_1 = flow_s2_f32['flow_12_1']
        flow_s2_f32_2 = flow_s2_f32['flow_12_2']
        flow_s2_f32_3 = flow_s2_f32['flow_12_3']
        flow_s2_f32_4 = flow_s2_f32['flow_12_4']
        img3_conv1, img3_conv2, img3_conv3, img3_conv4 = self.Encoderinput_img3_noise
        warp_s2_32_conv1 = self.Backward_warp(img3_conv1, flow_s2_f32_1)
        warp_s2_32_conv2 = self.Backward_warp(img3_conv2, flow_s2_f32_2)
        warp_s2_32_conv3 = self.Backward_warp(img3_conv3, flow_s2_f32_3)
        warp_s2_32_conv4 = self.Backward_warp(img3_conv4, flow_s2_f32_4)

        refSR_2 = self.UNet_decoder_4(img2_conv1, img2_conv2, img2_conv3, img2_conv4, warp_s2_12_conv1,warp_s2_12_conv2, warp_s2_12_conv3,warp_s2_12_conv4,warp_s2_32_conv1,warp_s2_32_conv2, warp_s2_32_conv3,warp_s2_32_conv4)
        return refSR_2


# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_Original(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(Crossnetpp_Original, self).__init__()

        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff, require_flow=False, flow_visible=False):
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        flow_s1_12_1 = flow_s1['flow_12_1']

        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')
        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        # sr_img = np.clip(refSR_2.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/sr_img.png')
        # exit()
        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_Multiflow(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(Crossnetpp_Multiflow, self).__init__()

        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff, vimeo=False, require_flow=False, encoder_input='input_img1_SR', flow_visible=False,frame_num=7):

        if vimeo:
            input_LR = buff['input_LR'].cuda()
            input_img1_LR = buff['input_img1_LR'].cuda()
            input_img1_HR = buff['input_img1_HR'].cuda()
            input_img1_SR = buff['input_img1_SR'].cuda()
            input_img2_HR = buff['input_img2_HR'].cuda()
        else:
            input_LR = torch.from_numpy(buff['input_LR']).cuda()
            input_img1_LR = buff['input_img1_LR'].cuda()
            # input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
            input_img1_SR = torch.from_numpy(buff[encoder_input]).cuda()
            input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_1 = flow_s1['flow_12_1']
            input_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)

        warp_img2_HR = input_img2_HR

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_Multiflow2(nn.Module):

    def __init__(self):
        super(Crossnetpp_Multiflow2, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff, frame_num=30):

        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()

        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_12_1 = flow['flow_12_1']
            input_img2_HR = self.Backward_warp(input_img2_HR, flow_12_1)
            warp_img2_HR_list = warp_img2_HR_list + (input_img2_HR,)

        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)
        warp_img2_HR = input_img2_HR

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_MultiflowFusion(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, vimeo=True, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        for i in range(frame_num):
            flow_s1_12 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1_12['flow_12_1'],)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_residue = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
        flow_s1_final = flow_residue + flow_s1_12_list[-1]

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_MultiflowFusion2(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion2, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=30):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_residue = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
        flow_s1_final = flow_residue + flow_s1_12_list[-1]

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_MultiflowFusion3(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion3, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=30):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :], input_img2_HR)
            flow_s1_12_1 = flow_s1['flow_12_1']
            input_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
            flow_s1_12_list = flow_s1_12_list + (flow_s1_12_1,)
            warp_img2_HR_list = warp_img2_HR_list + (input_img2_HR,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

# # 1st stage warped on input img, 2nd stage is crossnet
# class Crossnetpp_MultiflowFusion3(nn.Module):
#     def __init__(self):
#         super(Crossnetpp_MultiflowFusion3, self).__init__()
#         self.FlowNet_s1 = FlowNet(6)
#         self.FlowNet_s2 = FlowNet(6)
#         self.Backward_warp = Backward_warp()
#         self.Encoder = Encoder(3)
#         self.UNet_decoder_2 = UNet_decoder_2()
#
#         self.Flow_Encoder = Encoder(14)   # 14 = 2*7
#         self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)
#
#     def forward(self, buff, flow_visible=False, vimeo=True, require_flow=False, frame_num=7):
#         input_LR = buff['input_LR'].cuda()
#         input_img1_LR = buff['input_img1_LR'].cuda()
#         input_img1_HR = buff['input_img1_HR'].cuda()
#         input_img1_SR = buff['input_img1_SR'].cuda()
#         input_img2_HR = buff['input_img2_HR'].cuda()
#
#         flow_s1_12_list = tuple()
#         warp_img2_HR_list = tuple()
#         for i in range(frame_num):
#             flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :], input_img2_HR)
#             flow_s1_12_1 = flow_s1['flow_12_1']
#             input_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
#             flow_s1_12_list = flow_s1_12_list + (flow_s1_12_1,)
#             warp_img2_HR_list = warp_img2_HR_list + (input_img2_HR,)
#         warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)
#
#         ########## flow fusion
#         flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
#         # print(flow_12_list[0].size(), flow_12_list.size())
#         flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
#         flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
#
#         ########## warp
#         warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)
#
#         flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
#         flow_s2_12_1 = flow_s2['flow_12_1']
#         flow_s2_12_2 = flow_s2['flow_12_2']
#         flow_s2_12_3 = flow_s2['flow_12_3']
#         flow_s2_12_4 = flow_s2['flow_12_4']
#
#         ######### CrossNet
#         SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
#         HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)
#
#         warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
#         warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
#         warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
#         warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
#
#         refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
#                                       warp_s2_21_conv3, warp_s2_21_conv4)
#
#         if flow_visible:
#             return warp_img2_HR, refSR_2, flow_s1, flow_s2
#         else:
#             if require_flow:
#                 return warp_img2_HR, refSR_2, flow_s1_12_1
#             else:
#                 return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_MultiflowFusion4(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion4, self).__init__()
        self.FlowNet_LR = FlowNet(6)
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=30):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_LR_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1 = flow_s1['flow_12_1']
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        flow_LR_list = flow_LR_list + (flow_s1_12_1,)
        for i in range(frame_num - 1):
            flow_LR = self.FlowNet_LR(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR = self.Backward_warp(warp_img2_HR, flow_LR_1)
            flow_LR_list = flow_LR_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_LR_list_tensor = torch.cat(flow_LR_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_LR_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)
        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

# 1st stage warped on input img, 2nd stage is crossnet
class Crossnetpp_MultiflowFusion5(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion5, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7): # frame_num=30 for MPII dataset
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = input_img1_LR
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor