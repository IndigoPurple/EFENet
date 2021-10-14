import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from FlowNet_model import FlowNet
from Backward_warp_layer import Backward_warp
import matplotlib.pyplot as plt

class SupervisedFlowNet(nn.Module):

    def __init__(self):
        super(SupervisedFlowNet, self).__init__()

        self.FlowNet_fine = FlowNet(6)
        for p in self.parameters():
            p.requires_grad=False
        self.FlowNet_coarse = FlowNet(6)

    def forward(self, buff):

        input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        flow_fine = self.FlowNet_fine(input_img1_HR,input_img2_HR)
        flow_coarse = self.FlowNet_coarse(input_img1_LR, input_img2_HR)
        

        return flow_coarse,flow_fine
