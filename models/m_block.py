import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        elif conv_type == 'common':
            self.shift_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class FD(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(FD, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


class LGAN(nn.Module):
    def __init__(self, channels, window_size=5):
        super(LGAN, self).__init__()
        self.window_size = window_size
        self.split_chns = [channels * 2 // 3 for _ in range(3)]
        self.project_inp = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1),
                                         nn.BatchNorm2d(channels * 2))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        wsize = self.window_size
        ys = []
        # window attention
        q, v = rearrange(
            xs[0], 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=wsize, dw=wsize
        )
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(
            y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
        )
        ys.append(y_)
        # shifted window attention
        x_ = torch.roll(xs[1], shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
        q, v = rearrange(
            x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=wsize, dw=wsize
        )
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(
            y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
        )
        y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
        ys.append(y_)
        # long-range attentin
        # for row
        q, v = rearrange(xs[2], 'b (qv c) h w -> qv (b h) w c', qv=2)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column
        q = rearrange(q, '(b h) w c -> (b w) h c', b=b)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(y_, '(b w) h c-> b c h w', b=b)
        ys.append(y_)

        y = torch.cat(ys, dim=1)
        y = self.project_out(y)
        return y


class FEB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, window_size=5, act_type='gelu'):
        super(FEB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.FD = FD(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, act_type=act_type)
        self.LGAN = LGAN(channels=inp_channels, window_size=window_size)

    def forward(self, x):
        x = self.LGAN(x) + x
        x = self.FD(x) + x
        return x