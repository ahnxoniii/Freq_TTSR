import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from model import MainNet, LTE, SearchTransfer, TTSR


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*3, n_feats)
        self.conv_merge2 = conv3x3(n_feats*3, n_feats)
        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x
    


class Fre_fuse_1(nn.Module):

    def __init__(self, n_feats):
        super(Fre_fuse_1,self).__init__()
        self.n_feats = n_feats
        self.Fuse = SearchTransfer.Fuse()

    def forward(self,lr_fre=None, T_fre=None, lr_lv3_fre_CbCr=None):

        split_lr_fre = torch.unbind(lr_fre, dim=1)  # 리스트 (256개), torch.Size([9, 40, 40])
        split_T_fre = torch.unbind(T_fre, dim=1)  # 리스트 (256개),  torch.Size([9, 40, 40])

        # 2. Fuse 결과 저장할 리스트
        fused_results = []
        x_results = []

        # 3. zip을 사용하여 두 리스트의 요소를 쌍으로 반복
        for single_lr_fre, single_T_fre in zip(split_lr_fre, split_T_fre):

            single_lr_fre = single_lr_fre.unsqueeze(1)  #  torch.Size([9, 1, 40, 40])
            single_T_fre = single_T_fre.unsqueeze(1)  # torch.Size([9, 1, 40, 40])
            fused_result, x = self.Fuse(single_lr_fre, single_T_fre)  # 각각 대응하는 채널 입력
            fused_results.append(fused_result)
            x_results.append(x)
            
        # 4. 리스트를 다시 채널 차원에서 합치기
        fre_data_1= torch.stack(fused_results, dim=1)  # [B, 256, H, W]
        fre_data_1 = fre_data_1.squeeze(2)  
        x= torch.stack(x_results, dim=1)  # [B, 256, H, W]
        x = x.squeeze(2)  

        return fre_data_1, x


class MainNet(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MainNet, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        ### stage11
        self.conv11_head = conv3x3(256+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 2
        self.conv12 = conv3x3(n_feats, n_feats*4) #256
        self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        #self.conv21_head = conv3x3(n_feats, n_feats)
        self.conv22_head = conv3x3(128+n_feats, n_feats)

        self.ex12 = CSFI2(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        ### subpixel 2 -> 3
        self.conv23 = conv3x3(n_feats, n_feats*4)
        self.ps23 = nn.PixelShuffle(2)

        ### stage31, 32, 33
        #self.conv31_head = conv3x3(n_feats, n_feats)
        #self.conv32_head = conv3x3(n_feats, n_feats)
        self.conv33_head = conv3x3(64+n_feats, n_feats)

        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.merge_tail = MergeTail(n_feats)

        #--------------
        # self.Fuse = SearchTransfer.Fuse()
        # self.conv_fre = nn.Sequential(
        #     nn.Conv2d(8, n_feats, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        #self.Fre_fuse=Fre_fuse(n_feats)
        self.Fre_fuse_40=Fre_fuse_1(n_feats)
        self.Fre_fuse_80=Fre_fuse_1(n_feats)
        self.Fre_fuse_160=Fre_fuse_1(n_feats)
        self.conv_fre = conv1x1(64, 256)
        self.conv_fre2 = conv1x1(64, 128)
        self.conv_fre_40 = conv1x1(256, 64)
        self.conv_fre_spa_40 = conv3x3(128,128)
        self.conv_fre_spa_40_11 = conv1x1(128, 64)
        self.conv_fre_80 = conv1x1(128, 64)
        self.conv_fre_spa_80 = conv3x3(128,128)
        self.conv_fre_spa_80_11 = conv1x1(128, 64)
        self.conv_fre_spa_160 = conv3x3(128,128)
        self.conv_fre_spa_160_11 = conv1x1(128, 64)
        #--------------


    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None, 
               lrsr_lv3=None, lrsr_lv2=None, lrsr_lv1=None):
        ### shallow feature extraction
        x = self.SFE(x) #torch.Size([2, 64, 40, 40])
        
        #---------------------------------
        x_fre = self.conv_fre(x) #torch.Size([2, 64->256, 40, 40])
        fre_data_1, ifft_data_3 = self.Fre_fuse_40(x_fre, T_lv3)  # input : 256, 256 #torch.Size([2, 256, 40, 40])
        
        #---------------------------------
        ### stage11
        x11 = x #torch.Size([2, 64, 40, 40])
        ### soft-attention
        x11_res = x11 #torch.Size([2, 64, 40, 40])
        #T_lv3 shape torch.Size([2, 256, 40, 40])
        
        x11_res = torch.cat((x11_res, T_lv3), dim=1) #torch.Size([2, 64 + 256 , 40, 40]) 
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))  #torch.Size([2, 64, 40, 40]) 여기서 줄여짐
        x11_res = x11_res * S #torch.Size([2, 64, 40, 40])
        fre40 = self.conv_fre_40(fre_data_1) #torch.Size([2, 64, 40, 40])
        fre40 = fre40 * S #torch.Size([2, 64, 40, 40])
        fre_spa_40 = torch.cat((x11_res, fre40), dim=1) # 64+64 
        fre_spa_40 = self.conv_fre_spa_40(fre_spa_40) # 128 -> 128
        fre_spa_40 = self.conv_fre_spa_40_11(fre_spa_40) # 128-> 64
        
        x11 = x11 + fre_spa_40 #torch.Size([2, 64, 40, 40])

        x11_res = x11 #torch.Size([2, 64, 40, 40])

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res #torch.Size([2, 64, 40, 40])
        ### stage21, 22
        x21 = x11
        x21_res = x21
        x22 = self.conv12(x11) #torch.Size([2, 256, 40, 40])
        x22 = F.relu(self.ps12(x22)) #torch.Size([2, 64, 80, 80])
       

        ### soft-attention
        #---------------------------------
        x22_fre = self.conv_fre2(x22) #torch.Size([2, 64 -> 128, 80, 80])
        fre_data_1, ifft_data_2 = self.Fre_fuse_80(x22_fre, T_lv2)  #torch.Size([2, 128, 80, 80])
         
        #---------------------------------

        x22_res = x22 #torch.Size([2, 64, 80, 80])
        
        x22_res = torch.cat((x22_res, T_lv2), dim=1) #torch.Size([2, 64 + 128, 80, 80])
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res)) #torch.Size([2, 64, 80, 80])
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic') #torch.Size([2, 64, 80, 80])

        fre_data_1 = self.conv_fre_80(fre_data_1) #torch.Size([2, 128 -> 64, 80, 80])
        fre80 = fre_data_1 * F.interpolate(S, scale_factor=2, mode='bicubic') #torch.Size([2, 64, 80, 80])
        fre_spa_80 = torch.cat((x22_res, fre80), dim=1) # 64 + 64
        fre_spa_80 = self.conv_fre_spa_80(fre_spa_80) # 128 -> 128
        fre_spa_80 = self.conv_fre_spa_80_11(fre_spa_80) # 128-> 64
        x22 = x22 + fre_spa_80

        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = self.conv23(x22)
        
        x33 = F.relu(self.ps23(x33)) #torch.Size([8, 64, 160, 160])
        ### soft-attention

        #---------------------------------

        fre_data_1, ifft_data_1 = self.Fre_fuse_160(x33, T_lv1)  #torch.Size([2, 64, 160, 160])
         
        #---------------------------------

        x33_res = x33 #torch.Size([2, 64, 160, 160])
        
        x33_res = torch.cat((x33_res, T_lv1), dim=1) #torch.Size([2, 64 + 64 + 64 , 160, 160])
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(S, scale_factor=4, mode='bicubic')

        fre160 = fre_data_1 * F.interpolate(S, scale_factor=4, mode='bicubic') #torch.Size([2, 64, 160, 160])
        fre_spa_160 = torch.cat((x33_res, fre160), dim=1) # 64 + 64
        fre_spa_160 = self.conv_fre_spa_160(fre_spa_160) # 128 -> 128
        fre_spa_160 = self.conv_fre_spa_160_11(fre_spa_160) # 128-> 64
        x33 = x33 + fre_spa_160
        
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x = self.merge_tail(x31, x32, x33)
        
        return x, ifft_data_3, ifft_data_2, ifft_data_1
