import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from MultiscaleWarpingNet import MultiscaleWarpingNet
from Backward_warp_layer import Backward_warp
import matplotlib.pyplot as plt

class SupervisedCrossNet(nn.Module):

    def __init__(self):
        super(SupervisedCrossNet, self).__init__()

        self.CrossNet_fine = MultiscaleWarpingNet()
        for p in self.parameters():
            p.requires_grad=False
        self.FlowNet_coarse = MultiscaleWarpingNet()

    def forward(self, buff):

        SR_fine = self.CrossNet_fine(buff,mode = 'input_img1_HR')
        SR_coarse = self.FlowNet_coarse(buff, mode = 'input_img2_HR')


        return SR_coarse,SR_fine

