import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))] #원하는 차원이 아닌 곳을 1로 두어서 자유롭게 변경
        expanse = list(input.size()) 
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index) #원하는 인덱스의 패치를 가져옴

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3): #sr은 바이큐빅으로 크기 맞춘 것 
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1) #제일 작은 Q을 3*3으로 펼침
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1) #제일 작은 K을 3*3으로 펼침
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1) # c가 뒤로 가게 함

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W] 코사인 유사도 계산
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W] max 유사도와 유사도 인덱스 (S,H)

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1
    
### -------------------------------------------------------------

def fft(input):
    '''
    input: tensor of shape (batch_size, 1, height, width)
    mask: tensor of shape (height, width)
    '''
    # 执行2D FFT
    img_fft = torch.fft.rfftn(input, dim=(-2, -1))
    amp = torch.abs(img_fft)
    pha = torch.angle(img_fft)
    return amp, pha


class Att_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        att = self.att(x)
        x = x * att
        return x


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.convx = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class DMRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ir_embed = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.vi_embed = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.ir_att1 = Att_Block(out_channels, out_channels)
        self.ir_att2 = Att_Block(out_channels, out_channels)
        self.vi_att1 = Att_Block(out_channels, out_channels)
        self.vi_att2 = Att_Block(out_channels, out_channels)
        self.grad_ir = Sobelxy(out_channels)
        self.grad_vi = Sobelxy(out_channels)

    def forward(self, x, y):
        x = self.ir_embed(x)
        y = self.vi_embed(y)
        # return x, y
        t = x + y
        x1 = self.ir_att1(x)
        y1 = self.vi_att1(y)
        x2 = self.ir_att2(t)
        y2 = self.vi_att2(t)
        ir_grad = self.grad_ir(x)
        vi_grad = self.grad_vi(y)
        return x1 + x2 + ir_grad, y1 + y2 + vi_grad


class Fuse_block(nn.Module):
    def __init__(self, dim, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim, channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.down_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()),
        )

    def forward(self, ir, vi, frefus):
        x = torch.cat([ir, vi, frefus], dim=1)  # n,c,h,w
        x = self.encoder(x)
        x = self.down_conv(x)
        return x


class IFFT(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),)
        self.press = nn.Conv2d(out_channels, 1, kernel_size=1)  

    def forward(self, amp, pha):
        real = amp * torch.cos(pha) + 1e-8
        imag = amp * torch.sin(pha) + 1e-8
        x = torch.complex(real, imag)
        #print("before ifft x:",x.shape)
        x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
        #print("after ifft x:",x.shape)
        x = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        #print("x2:",x.shape)
        x = self.conv1(x)
        x = self.press(x)
        #print("last x:",x.shape)
        return x


class AmpFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = self.conv1(x)
        return x


class PhaFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = self.conv1(x)
        return x


class Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 8
        #self.dmrm = DMRM(1, self.channel)
        self.ff1 = AmpFuse()
        self.ff2 = PhaFuse()
        self.ifft = IFFT(self.channel)
        #self.fus_block = Fuse_block(self.channel * 3)

    def forward(self, ir, vi):
        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)
        amp = self.ff1(ir_amp, vi_amp)
        pha = self.ff2(ir_pha, vi_pha)
        frefus = self.ifft(amp, pha)
        
        real = amp * torch.cos(pha) + 1e-8
        imag = amp * torch.sin(pha) + 1e-8
        x = torch.complex(real, imag)
        x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))

        return frefus, x