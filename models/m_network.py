import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.m_block import FEB, MeanShift


def create_model(args):
    return LGAN(args)


class LGAN(nn.Module):
    def __init__(self, args):
        super(LGAN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.distill_number = 2 * len(self.window_sizes)
        self.n_lgab = args.n_lgab
        self.c_lgan = args.c_lgan
        self.r_expand = args.r_expand
        self.act_type = args.act_type

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_lgan, kernel_size=3, stride=1, padding=1)]
        # define body module
        self.body = nn.ModuleList()
        number = self.n_lgab // len(self.window_sizes)
        for i in range(number):
            for window_size in self.window_sizes:
                self.body.append(FEB(self.c_lgan, self.c_lgan, self.r_expand, window_size, act_type=self.act_type))

        # define tail module
        m_tail = [
            nn.Conv2d(self.c_lgan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, train=False):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for idx, stage in enumerate(self.body):
            res = stage(res)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load(self, state_dict, strict=True, compatibility=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            #         compatibility mode
            if compatibility:
                name = name.replace('modules_lfe.lfe_0', 'FD').replace('modules_gmsa.gmsa_0', 'LGAB')
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    # own_state[name].requires_grad = False
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass
