from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.conv_128 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_64 = nn.Conv2d(256, 64, kernel_size=1)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        lrsr_lv1, lrsr_lv2, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.) #detach는 그래디언트를 분리해줌
        #_, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.) #detach는 그래디언트를 분리해줌
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)
        #--------------
        #lrsr_lv2  = F.interpolate(lrsr_lv3, scale_factor=2, mode='bicubic')
        #lrsr_lv1  = F.interpolate(lrsr_lv2, scale_factor=2, mode='bicubic')
        #--------------
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)
        

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        #lrsr_lv2 = self.conv_128(lrsr_lv2)
        #lrsr_lv1 = self.conv_64(lrsr_lv1)
        # lrsr_lv3 : (B, 256, H/4, W/4)
        # lrsr_lv2 : (B, 256, H/2, W/2) -> 128
        # lrsr_lv1 : (B, 256, H, W) -> 64
        # T_lv3 : (B, 256, H/4, W/4)
        # T_lv2 : (B, 128, H/2, W/2)
        # T_lv1 : (B, 64, H, W)
        
        
        sr,ifft_data_3, ifft_data_2, ifft_data_1 = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1, lrsr_lv3, lrsr_lv2, lrsr_lv1) ##

        return sr, S, T_lv3, T_lv2, T_lv1, 
            
        #ifft_data_3, ifft_data_2, ifft_data_1, lrsr_lv3, lrsr_lv2, lrsr_lv1