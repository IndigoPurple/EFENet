import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from FlowNet_model import FlowNet
from Backward_warp_layer import Backward_warp

class MWNet(nn.Module):

    def __init__(self):
        super(MWNet, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()


    def forward(self, input_img1_LR,input_img1_SR,input_img2_HR,encoder_mode = 'SR', return_flow = False):

        flow = self.FlowNet(input_img1_LR, input_img2_HR)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']
        
        if encoder_mode == 'SR':
            SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        elif encoder_mode == 'LR':
            SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_LR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)
      
        sythsis_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)
 
        if return_flow == True:
            return sythsis_output,flow
        else:
            return sythsis_output


class MWNet_2Stage(nn.Module):


    def __init__(self):
        super(MWNet_2Stage, self).__init__()
        self.MWNet_coarse = MWNet()
        self.MWNet_fine = MWNet()

    def forward(self, buff,encoder_mode = 'SR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        coarse_img1_SR = self.MWNet_coarse(input_img1_LR,input_img1_SR,input_img2_HR)

        fine_img1_SR = self.MWNet_fine(coarse_img1_SR,input_img1_SR,input_img2_HR,encoder_mode = encoder_mode)

        return coarse_img1_SR,fine_img1_SR


#fix 1st stage and use coarse_sr as both flow and SR input.
#The shared encoder prefers to filling the resolution gap between the LR and HR to make flow learning easier.
#I consider that as the resolution gap getting samller, feature maps may contains more high-frequency information.
class MWNet_2Stage_2(nn.Module):


    def __init__(self):
        super(MWNet_2Stage_2, self).__init__()
        self.MWNet_coarse = MWNet()
        #for p in self.parameters():
        #    p.requires_grad=False
        self.MWNet_fine = MWNet()

    def forward(self, buff,encoder_mode = 'SR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        coarse_img1_SR = self.MWNet_coarse(input_img1_LR,input_img1_SR,input_img2_HR)

        fine_img1_SR = self.MWNet_fine(coarse_img1_SR,input_img1_SR,input_img2_HR,encoder_mode = encoder_mode)

        return coarse_img1_SR,fine_img1_SR

#Using HR&HR flownet to supervise the 2nd stage refSR&HR flow
class MWNet_2Stage_3(nn.Module):


    def __init__(self):
        super(MWNet_2Stage_3, self).__init__()
        self.FlowNet_HR = FlowNet(6)
        self.MWNet_coarse = MWNet()
        for p in self.parameters():
            p.requires_grad = False
        self.MWNet_fine = MWNet()
 
    def forward(self, buff,encoder_mode = 'SR'):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
     
        coarse_img1_SR = self.MWNet_coarse(input_img1_LR,input_img1_SR,input_img2_HR)

        fine_img1_SR,flow_s2 = self.MWNet_fine(coarse_img1_SR,input_img1_SR,input_img2_HR,encoder_mode = encoder_mode,return_flow = True)

        flow_HR = self.FlowNet_HR(input_img1_HR,input_img2_HR)

        return coarse_img1_SR,fine_img1_SR, flow_HR, flow_s2





if __name__ == '__main__':
    net = MWNet_2Stage().state_dict()
    for k,v in net.items():
        print(k)


