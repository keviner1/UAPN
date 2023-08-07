from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
CE = torch.nn.BCELoss(reduction='sum')
import numpy as np
import torch.nn.init as init
import sys
sys.path.append("..")
from Registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class UAPN(nn.Module):
    def __init__(self, channels):
        super(UAPN, self).__init__()

        self.conv_ms = nn.Conv2d(4, channels, 1, 1, 0)
        self.conv_pan = nn.Conv2d(1, channels, 1, 1, 0)
        self.conv_out = nn.Conv2d(channels, 4, 1, 1, 0)

        self.FEM1 = Unet(channels)                   
        self.FEM2 = Unet(channels)  

        self.UAFM1 = UAFM(channels)
        self.UAFM2 = UAFM(channels)

        self.skip1 = skip(channels*2, channels, "CONV")
        self.skip2 = skip(channels*2, channels, "CONV")


        self.apply(initialize_weights)
        
    def forward(self, pan, lms):
        ms_0 = F.interpolate(lms, scale_factor=4, mode='bicubic', align_corners=True)
        Fin = self.conv_ms(ms_0)
        panf = self.conv_pan(pan)
        #-----------------------
        F1 = self.FEM1(Fin)
        F1 = self.skip1( torch.cat([Fin, F1],1),  F1)
        AU1, EU1, mean1, F1 = self.UAFM1(F1, ms_0, panf)

        F2 = self.FEM2(F1)
        F2 = self.skip2( torch.cat([F1, F2],1),  F2)
        AU2, EU2, mean2, F2 = self.UAFM2(F2, ms_0, panf)
        
        out = self.conv_out(F2) + ms_0
        return [AU1, AU2], [EU1, EU2], [mean1, mean2], out


class UAFM(nn.Module):
    def __init__(self, channels):
        super(UAFM, self).__init__()
        self.condition_channels = 4 #AU, EU, Gpan
        self.UE = UncertaintyEstimation(channels = channels, T = 10, q = 0.2)
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias = True), nn.LeakyReLU(negative_slope=0.2, inplace=True)) #nn.InstanceNorm2d(channels),
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias = True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.UGAConv1 = UAConv(self.condition_channels, channels*2, channels, 3, stride=1, padding=1, dilation=1, groups=1, use_bias=True)
        self.UGAConv2 = UAConv(self.condition_channels, channels*2, channels, 3, stride=1, padding=1, dilation=1, groups=1, use_bias=True)

    def forward(self, F_1, ms, pan):
        AU, EU, mean = self.UE(F_1, ms)
        F_2 = self.conv1(self.UGAConv1(torch.cat([F_1,pan],1), EU))
        F_out = self.conv2(self.UGAConv2(torch.cat([F_1,F_2],1), 1-EU))
        return AU, EU, mean, F_out

#Uncertainty_Estimation
class UncertaintyEstimation(nn.Module):
    def __init__(self, channels, T, q):
        super(UncertaintyEstimation, self).__init__()
        self.T = T
        self.q = q
        self.conv = nn.Sequential(nn.Conv2d(channels, channels*2, 3, 1, 1, bias = True))
        self.out = nn.Sequential(nn.Conv2d(channels*2, 4, 1, 1, 0), nn.Tanh())
        self.aue = nn.Sequential(nn.Conv2d(channels*2, 4, 3, 1, 1), nn.Sigmoid())

    def random_mask(self, x, q):
        mask = np.random.binomial(n=1, p=1-q, size=x.shape[1])
        mask = torch.tensor(mask).cuda()
        mask = rearrange(mask, "C -> 1 C 1 1")
        return x * mask

    def epistemic_uncetainty(self, x, lms):
        mean = 0
        xs = []
        for i in range(self.T):
            x_cur = self.out(self.random_mask(x, self.q)) + lms
            x_cur = rearrange(x_cur, "B C H W -> 1 B C H W")
            xs.append(x_cur)
        
        xs = torch.cat(xs, dim=0)
        EU, mean = torch.var_mean(input = xs, dim = 0, unbiased = True)
        return EU, mean

    def aleatoric_uncetainty(self, x):
        return self.aue(x)

    def forward(self, x, lms):
        x = self.conv(x)
        EU, mean = self.epistemic_uncetainty(x, lms)
        AU = self.aleatoric_uncetainty(x)
        return AU, EU, mean


class UAConv(nn.Module):
    """
     The UAConv code writing here draws on LAGConv (https://github.com/liangjiandeng/LAGConv)
    """
    def __init__(self, condition_channels, channels_in, channels_out, kernel_size, stride=1, padding=1, dilation=1, groups=1, use_bias=True):
        super(UAConv, self).__init__()
        self.condition_channels = condition_channels
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        self.spatial_generator=nn.Sequential(
            nn.Conv2d(condition_channels, kernel_size**2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size**2, kernel_size**2, 1),
            nn.ReLU(),
            nn.Conv2d(kernel_size**2, kernel_size**2, 1),
            nn.Sigmoid()
        )
        self.spectral_generator=nn.Sequential(
                nn.Conv2d(condition_channels, channels_in, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in, channels_in, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in,channels_in, 1, 1, 0),
                nn.Sigmoid()
        )
        conv1=nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, dilation, groups, use_bias)
        self.weight=conv1.weight # m, n, k, k

    def forward(self,x, U):
        (b, n, H, W) = x.shape
        m=self.channels_out
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        #---------------spectral--------------------
        A_spectral=self.spectral_generator(U)
        A_spectral=A_spectral.permute([0,2,3,1])
        A_spectral=A_spectral.unsqueeze(-1).repeat([1,1,1,1,k*k])
        A_spectral=A_spectral.view(b,n_H,n_W,n*k*k)

        #--------------spatial---------------------
        A_spatial=self.spatial_generator(U)
        A_spatial=A_spatial.permute([0,2,3,1])
        A_spatial=A_spatial.repeat([1,1,1,n])

        #-------------A_ss-------------
        A_ss = A_spectral * A_spatial 
        A_ss=A_ss.view(b,n_H*n_W,n*k*k)
        A_ss=A_ss.permute([0,2,1]) 

        x_unfold=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) 
        x_reweighted=A_ss*x_unfold 
        x_reweighted=x_reweighted.permute([0,2,1]) 
        x_reweighted=x_reweighted.view(1,b*n_H*n_W,n*k*k) 

        Conv_w=self.weight.view(m,n*k*k) 
        Conv_w=Conv_w.permute([1,0]) 
        y=torch.matmul(x_reweighted,Conv_w) 
        y=y.view(b,n_H*n_W,m) 
        y=y.permute([0,2,1]) 
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1)
        return y

class skip(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(skip, self).__init__()
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=True),
                                        nn.InstanceNorm2d(channels_out, affine=True),nn.ReLU(inplace = True),)
        if block == "ID":
            self.body = nn.Identity()
        if block == "HIN":
            self.body = nn.Sequential(HinBlock(channels_in, channels_out))
        if block == "INV":
            self.body = nn.Sequential(InvBlock(channels_in, channels_in//2), nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=True),)
        #--------------------------------------
        self.alpha1 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha1.data.fill_(1.0)
        self.alpha2 = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha2.data.fill_(0.5)

    def forward(self, x, y):
        out = self.alpha1 * self.body(x) + self.alpha2 * y
        return out

class sample_block(nn.Module):
    def __init__(self, channels_in, channels_out, size, dil):
        super(sample_block, self).__init__()
        #------------------------------------------
        if size == "DOWN":
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )
        if size == "UP":
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, 3, 1, dil, dilation = dil),
                nn.InstanceNorm2d(channels_out, affine=True),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.conv(x)

class basic_block(nn.Module):
    def __init__(self, channels_in, channels_out, block):
        super(basic_block, self).__init__()
        #------------------------------------------
        if block == "CONV":
            self.body = nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=True),
                                        nn.InstanceNorm2d(channels_out, affine=True),nn.ReLU(inplace = True),)
        if block == "INV":
            self.body = nn.Sequential(InvBlock(channels_in, channels_out//2))
        if block == "HIN":
            self.body = nn.Sequential(HinBlock(channels_in, channels_out))

    def forward(self, x):
        return self.body(x)


class Unet(nn.Module):
    def __init__(self, channels):
        super(Unet, self).__init__()
        #---------ENCODE
        self.layer_dowm1 = basic_block(channels,channels,"INV")
        self.dowm1 = sample_block(channels, channels, "DOWN", 2)
        self.layer_dowm2 = basic_block(channels,channels,"INV")
        self.dowm2 = sample_block(channels, channels, "DOWN", 4)
        #---------DECODE
        self.layer_bottom = basic_block(channels,channels,"INV")
        self.up2 = sample_block(channels, channels, "UP", 4)
        self.layer_up2 = basic_block(channels,channels,"INV")
        self.up1 = sample_block(channels, channels, "UP", 2)
        self.layer_up1 = basic_block(channels,channels,"INV")
        #---------SKIP
        self.fus2 = skip(channels*2, channels, "HIN")
        self.fus1 = skip(channels*2, channels, "HIN")
        #---------SKIP
        self.skip1 = skip(channels*2, channels, "CONV")
        self.skip2 = skip(channels*2, channels, "CONV")
        self.skip3 = skip(channels*2, channels, "CONV")
        self.skip4 = skip(channels*2, channels, "CONV")
        self.skip5 = skip(channels*2, channels, "CONV")
        self.skip6 = skip(channels*2, channels, "CONV")

    def forward(self, x):
        #---------ENCODE
        x_11 = self.layer_dowm1(x)
        x_down1 = self.dowm1(x_11)
        x_down1 = self.skip1(torch.cat([x,x_down1],1), x_down1)

        x_12 = self.layer_dowm2(x_down1)
        x_down2 = self.dowm2(x_12)
        x_down2 = self.skip2(torch.cat([x_down1,x_down2],1), x_down2)
        x_down2 = self.skip3(torch.cat([x,x_down2],1), x_down2)

        x_bottom = self.layer_bottom(x_down2)

        #---------DECODE
        x_up2 = self.up2(x_bottom)
        x_up2 = self.skip4(torch.cat([x_bottom,x_up2],1), x_up2)
        x_22 = self.layer_up2(x_up2)
        x_22 = self.fus2(torch.cat([x_12,x_22],1), x_22)

        x_up1 = self.up1(x_22)
        x_up1 = self.skip5(torch.cat([x_22,x_up1],1), x_up1)
        x_21 = self.layer_up1(x_up1)
        x_21 = self.skip6(torch.cat([x_bottom,x_21],1), x_21)
        x_21 = self.fus1(torch.cat([x_11,x_21],1), x_21)
    
    
        return x_21


class HinBlock(nn.Module):
    def __init__(self,in_size,out_size):
        super(HinBlock,self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride = 1, padding=1, bias=True)
        self.relu_1 = nn.Sequential( nn.LeakyReLU(0.2, inplace=False), )
        self.conv_2 = nn.Sequential( nn.Conv2d(out_size, out_size, kernel_size=3, stride = 1, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=False),)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out += self.identity(x)
        return out 

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'HIN':
            return HinBlock(channel_in, channel_out)

    return constructor

class InvBlock(nn.Module):
    def __init__(self,channel_num, channel_split_num, subnet_constructor = subnet('HIN'), clamp=0.8):   ################  split_channel一般设为channel_num的一半
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x):

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out+x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)