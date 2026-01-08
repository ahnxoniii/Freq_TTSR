from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        loss = F.mse_loss(sr_relu5_1, hr_relu5_1)
        return loss


class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def gram_matrix(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, h*w)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        ### S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*2, S.size(-1)*2), mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*4, S.size(-1)*4), mode='bicubic'))

            print("S_lv3:", S_lv3.min(), S_lv3.max(), S_lv3.mean())
            print("S_lv2:", S_lv2.min(), S_lv2.max(), S_lv2.mean())
            print("S_lv1:", S_lv1.min(), S_lv1.max(), S_lv1.mean())
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture  = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            print("")
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
            
        elif (self.type == 'l2'):
            loss_v3  = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_v2  = F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_v1  = F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture = loss_v3 + loss_v2 + loss_v1
            loss_texture /= 3.
            # with open('tpl_loss.txt', 'a') as f:  # 'a'는 append 모드(기존 내용에 추가) 
            #     f.write(f"loss_texture (final): {loss_texture.item()}\n")
            #     f.write(f"loss_lv3: {loss_v3.item()}\n")
            #     f.write(f"loss_lv2: {loss_v2.item()}\n")
            #     f.write(f"loss_lv1: {loss_v1.item()}\n")
            #     f.write(f"map_lv3 * S_lv3: {(map_lv3 * S_lv3).min()} {(map_lv3 * S_lv3).max()}\n")
            #     f.write(f"T_lv3 * S_lv3: {(T_lv3 * S_lv3).min()} {(T_lv3 * S_lv3).max()}\n")
            #     f.write(f"map_lv2 * S_lv2: {(map_lv2 * S_lv2).min()} {(map_lv2 * S_lv2).max()}\n")
            #     f.write(f"T_lv2 * S_lv2: {(T_lv2 * S_lv2).min()} {(T_lv2 * S_lv2).max()}\n")
            #     f.write(f"map_lv1 * S_lv1: {(map_lv1 * S_lv1).min()} {(map_lv1 * S_lv1).max()}\n")
            #     f.write(f"T_lv1 * S_lv1: {(T_lv1 * S_lv1).min()} {(T_lv1 * S_lv1).max()}\n")
            #     f.write(f"map_lv3: {map_lv3.min()} {map_lv3.max()} {map_lv3.mean()}\n")
            #     f.write(f"map_lv2: {map_lv2.min()} {map_lv2.max()} {map_lv2.mean()}\n")
            #     f.write(f"map_lv1: {map_lv1.min()} {map_lv1.max()} {map_lv1.mean()}\n")
            #     f.write(f"S: {S.min()} {S.max()} {S.mean()}\n")
            #     f.write(f"T_lv3: {T_lv3.min()} {T_lv3.max()} {T_lv3.mean()}\n")
            #     f.write(f"T_lv2: {T_lv2.min()} {T_lv2.max()} {T_lv2.mean()}\n")
            #     f.write(f"T_lv1: {T_lv1.min()} {T_lv1.max()} {T_lv1.mean()}\n")
        
        return loss_texture


class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1, 
        lr_dis=1e-4, train_crop_size=40):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(train_crop_size*4).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])
            
    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g
  
    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict


#---------------------------------------------------------------
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def cc(self, img1, img2):
        eps = torch.finfo(torch.float32).eps
        """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
        N, C, _, _ = img1.shape
        img1 = img1.reshape(N, C, -1)
        img2 = img2.reshape(N, C, -1)
        img1 = img1 - img1.mean(dim=-1, keepdim=True)
        img2 = img2 - img2.mean(dim=-1, keepdim=True)
        cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
        cc = torch.clamp(cc, -1.0, 1.0)
        return cc.mean()
    
    def forward(self, ifft_data_3, ifft_data_2, ifft_data_1, lrsr_lv3, lrsr_lv2, lrsr_lv1):
        # real = amp * torch.cos(pha) + 1e-8
        # imag = amp * torch.sin(pha) + 1e-8
        # x = torch.complex(real, imag)
        # x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))
        loss_fre_3 = self.cc(ifft_data_3,lrsr_lv3)
        loss_fre_2 = self.cc(ifft_data_2,lrsr_lv2)
        loss_fre_1 = self.cc(ifft_data_1,lrsr_lv1)
        # loss_ir = self.cc(x * mask, hr * mask)
        # loss_vi = self.cc(x * (1 - mask), hr * (1 - mask))
        loss = loss_fre_1 + loss_fre_2 + loss_fre_3
        #loss = (1 - loss_fre_3) + (1 - loss_fre_2) + (1 - loss_fre_1)
        return loss
    

# class AMPLoss(nn.Module):
#     def __init__(self):
#         super(AMPLoss, self).__init__()
#         self.cri = nn.L1Loss()

#     def forward(self, x, y):
#         x = torch.fft.rfft2(x, norm='backward')
#         x_mag =  torch.abs(x)
#         y = torch.fft.rfft2(y, norm='backward')
#         y_mag = torch.abs(y)

#         return self.cri(x_mag,y_mag) # x : SR , y : HR


# class PhaLoss(nn.Module):
#     def __init__(self):
#         super(PhaLoss, self).__init__()
#         self.cri = nn.L1Loss()

#     def forward(self, x, y):
#         x = torch.fft.rfft2(x, norm='backward')
#         x_mag = torch.angle(x)
#         y = torch.fft.rfft2(y, norm='backward')
#         y_mag = torch.angle(y)

#         return self.cri(x_mag, y_mag)

class FrelvLoss(nn.Module):
    def __init__(self):
        super(FrelvLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y, return_both=False):
        x_fft = torch.fft.rfft2(x, norm='backward')
        y_fft = torch.fft.rfft2(y, norm='backward')
        x_amp = torch.abs(x_fft)
        y_amp = torch.abs(y_fft)
        x_pha = torch.angle(x_fft)
        y_pha = torch.angle(y_fft)
        amp_loss = self.cri(x_amp, y_amp)
        pha_loss = self.cri(x_pha, y_pha)
        if return_both:
            return amp_loss, pha_loss
        return amp_loss + pha_loss
#--------------------------------------------------------------

def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type='l1')
    if (abs(args.per_w - 0) > 1e-8):
        loss['per_loss'] = PerceptualLoss()
    if (abs(args.tpl_w - 0) > 1e-8):
        loss['tpl_loss'] = TPerceptualLoss(use_S=args.tpl_use_S, type=args.tpl_type)
    if (abs(args.adv_w - 0) > 1e-8):
        loss['adv_loss'] = AdversarialLoss(logger=logger, use_cpu=args.cpu, num_gpu=args.num_gpu, 
            gan_type=args.GAN_type, gan_k=args.GAN_k, lr_dis=args.lr_rate_dis,
            train_crop_size=args.train_crop_size)
    if (abs(args.fre_w - 0) > 1e-8):
        loss['fre_loss'] = FrelvLoss()    
        #loss['fre_loss'] = FrequencyLoss()    
    return loss

