import random
import sys
import torch
from torch import nn
import torch.nn.functional as F
sys.path.append('./net')
from graph import (Graph_Joint, Graph_Mid, Graph_Bodypart,
                   PoolJointToMid, PoolMidToBodypart,
                   UnpoolBodypartToMid, UnpoolMidToJoint)
from blocks import  (StgcnBlock, ResStgcnBlock, 
                     BPStyleNet, ResBPStyleNet)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        enc_in_dim = config['enc_in_dim']
        enc_nf = config['enc_nf']
        latent_dim = config['latent_dim']
        graph_cfg = config['graph']

        self.enc_content = Encoder_con(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.enc_style = Encoder_sty(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.dec = Decoder(self.enc_content.output_channels, enc_in_dim, 
                           latent_dim=latent_dim, graph_cfg=graph_cfg)
        
        self.apply(self._init_weights)
    
    def forward(self, xa, xb, phase='train'):       # x:  (N, C, V, T)
        # encode
        c_xa = self.enc_content(xa)     # [(n, c, 21, 4t), ..., (n, 4c, 5, t)] 
        c_xb = self.enc_content(xb)
        s_xa = self.enc_style(xa)   
        s_xb = self.enc_style(xb)

        # style mixing
        mixing_prob = 0.4 if phase == 'train' else 0.0
        s_mix, bdy_part_select = mixing_styles(s_xa, s_xb, mixing_prob)

        # decode
        xab = self.dec(c_xa[-1], s_mix[0][::-1], s_mix[1][::-1], s_mix[2][::-1], s_mix[3][::-1], s_mix[4][::-1])
        xaa = self.dec(c_xa[-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1])  # reconstruction
        xbb = self.dec(c_xb[-1], s_xb[::-1], s_xb[::-1], s_xb[::-1], s_xb[::-1], s_xb[::-1])  # reconstruction

        c_xab = self.enc_content(xab)
        xaba = self.dec(c_xab[-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1])
        if len(bdy_part_select) == 0:
            s_xab = self.enc_style(xab)        
            xabb = self.dec(c_xb[-1], s_xab[::-1], s_xab[::-1], s_xab[::-1], s_xab[::-1], s_xab[::-1])
        else:
            xabb = xb
        
        return xaa, xbb, xab, xaba, xabb
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class Encoder_con(nn.Module):
    def __init__(self, in_channels, 
                       channels, 
                       graph_cfg,
                       edge_importance_weighting=True):
        super().__init__()
        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])

        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)

        # build networks
        spatial_kernel_size_j = self.A_j.size(0)     # ex) subset K= 0, 1, 2 -> 3
        spatial_kernel_size_m = self.A_m.size(0)     # 2
        spatial_kernel_size_b = self.A_b.size(0)     # 2
        ks_joint = (7, spatial_kernel_size_j)
        ks_mid = (5, spatial_kernel_size_m)
        ks_bodypart = (5, spatial_kernel_size_b)
        ks_bottleneck = (3, spatial_kernel_size_b)

        self.from_mot = nn.Conv2d(in_channels, channels, (1, 1))

        # G1 level
        self.joint = StgcnBlock(channels,   # from motion channel
                                2*channels, 
                                kernel_size=ks_joint, 
                                stride=1,   # temporal conv stride     
                                norm='in',
                                activation='lrelu')
        channels *= 2
        
        # G2 level
        self.down_JointToMid = PoolJointToMid()
        self.down_temp1 = F.avg_pool2d
        self.mid = StgcnBlock(channels, 
                              2*channels, 
                              kernel_size=ks_mid, 
                              stride=1, 
                              norm='in',
                              activation='lrelu')
        channels *= 2
        
        # G3 level
        self.down_MidToBodypart = PoolMidToBodypart()
        self.down_temp2 = F.avg_pool2d
        self.bodypart = StgcnBlock(channels, 
                                   2*channels,  
                                   kernel_size=ks_bodypart, 
                                   stride=1,  
                                   norm='in',
                                   activation='lrelu')
        channels *= 2

        # bottleneck
        self.bottleneck = ResStgcnBlock(channels,
                                        channels, 
                                        kernel_size=ks_bottleneck, 
                                        stride=1, 
                                        norm='in',
                                        activation='lrelu')

        self.output_channels = channels

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
        else:
            self.edge_importance_j = 1
            self.edge_importance_m = 1
            self.edge_importance_b = 1
            self.edge_importance_bt = 1

    def forward(self, x):
        latents_features = []
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)->(N, C, T, V)

        # from mot
        x = self.from_mot(x)

        # G1
        x = self.joint(x, self.A_j * self.edge_importance_j)
        latents_features.append(x)

        # G2
        x = self.down_JointToMid(x)
        x = self.down_temp1(x, kernel_size=(2, 1))
        x = self.mid(x, self.A_m * self.edge_importance_m)
        latents_features.append(x)

        # G3
        x = self.down_MidToBodypart(x)
        x = self.down_temp2(x, kernel_size=(2, 1))
        x = self.bodypart(x, self.A_b * self.edge_importance_b)
        latents_features.append(x)

        # bottleneck
        x = self.bottleneck(x, self.A_b * self.edge_importance_bt)
        latents_features.append(x)

        return latents_features


class Encoder_sty(nn.Module):
    def __init__(self, in_channels, 
                       channels, 
                       graph_cfg,
                       edge_importance_weighting=True):
        super().__init__()
        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])

        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)

        # build networks
        spatial_kernel_size_j = self.A_j.size(0)     # ex) subset K= 0, 1, 2 -> 3
        spatial_kernel_size_m = self.A_m.size(0)    # 2
        spatial_kernel_size_b = self.A_b.size(0)    # 2
        ks_joint = (7, spatial_kernel_size_j)
        ks_mid = (5, spatial_kernel_size_m)
        ks_bodypart = (5, spatial_kernel_size_b)
        ks_bottleneck = (3, spatial_kernel_size_b)

        self.from_mot = nn.Conv2d(in_channels, channels, (1, 1))

        # G1 level
        self.joint = StgcnBlock(channels,
                                2*channels, 
                                kernel_size=ks_joint, 
                                stride=1,      # temporal conv stride     
                                norm='none',
                                activation='lrelu')
        channels *= 2
        
        # G2 level
        self.down_JointToMid = PoolJointToMid()
        self.down_temp1 = F.avg_pool2d
        self.mid = StgcnBlock(channels, 
                              2*channels, 
                              kernel_size=ks_mid, 
                              stride=1, 
                              norm='none',
                              activation='lrelu')
        channels *= 2
        
        # G3 level
        self.down_MidToBodypart = PoolMidToBodypart()
        self.down_temp2 = F.avg_pool2d
        self.bodypart = StgcnBlock(channels, 
                                   2*channels,  
                                   kernel_size=ks_bodypart, 
                                   stride=1,  
                                   norm='none',
                                   activation='lrelu')
        channels *= 2

        # bottleneck
        self.bottleneck = ResStgcnBlock(channels,
                                        channels, 
                                        kernel_size=ks_bottleneck, 
                                        stride=1, 
                                        norm='none',
                                        activation='lrelu')

        self.output_channels = channels

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
        else:
            self.edge_importance_j = 1
            self.edge_importance_m = 1
            self.edge_importance_b = 1
            self.edge_importance_bt = 1

    def forward(self, x):
        latents_features = []
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)->(N, C, T, V)

        # from mot
        x = self.from_mot(x)

        # G1
        x = self.joint(x, self.A_j * self.edge_importance_j)
        latents_features.append(x)

        # G2
        x = self.down_JointToMid(x)
        x = self.down_temp1(x, kernel_size=(2, 1))
        x = self.mid(x, self.A_m * self.edge_importance_m)
        latents_features.append(x)

        # G3
        x = self.down_MidToBodypart(x)
        x = self.down_temp2(x, kernel_size=(2, 1))
        x = self.bodypart(x, self.A_b * self.edge_importance_b)
        latents_features.append(x)

        # bottleneck
        x = self.bottleneck(x, self.A_b * self.edge_importance_bt)
        latents_features.append(x)

        return latents_features


class Decoder(nn.Module):
    def __init__(self, 
                 channels, 
                 out_channels,
                 latent_dim,
                 graph_cfg,
                 edge_importance_weighting=True):
        super().__init__()

        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])

        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)
        
        # build networks
        spatial_kernel_size_b = self.A_b.size(0)    # 2
        spatial_kernel_size_m = self.A_m.size(0)    # 2
        spatial_kernel_size_j = self.A_j.size(0)    # ex) subset K= 0, 1, 2 -> 3
        ks_bottleneck = (3, spatial_kernel_size_b)
        ks_bodypart = (5, spatial_kernel_size_b)
        ks_mid = (5, spatial_kernel_size_m)
        ks_joint = (7, spatial_kernel_size_j)

        # bottleneck
        self.bottleneck = ResBPStyleNet(latent_dim,   # style dim
                                        channels,     # input channel
                                        channels,     # output channel
                                        kernel_size=ks_bottleneck, 
                                        stride=1, 
                                        activation='lrelu')

        # G3
        self.bodypart = BPStyleNet(latent_dim,
                                   channels,
                                   channels//2,
                                   kernel_size=ks_bodypart, 
                                   stride=1,
                                   activation='lrelu')
        channels //= 2

        # G2
        self.up_BodypartToMid = UnpoolBodypartToMid()
        self.up_temp1 = F.interpolate        # (x, scale_factor=(2, 1), mode='nearest')
        self.mid = BPStyleNet(latent_dim,
                              channels, 
                              channels//2, 
                              kernel_size=ks_mid, 
                              stride=1,
                              activation='lrelu') 
        channels //= 2

        # G1
        self.up_MidToJoint = UnpoolMidToJoint()
        self.up_temp2 = F.interpolate
        self.joint = BPStyleNet(latent_dim,
                                channels,
                                channels//2, 
                                kernel_size=ks_joint, 
                                stride=1,
                                activation='lrelu')
        channels //= 2

        # to mot
        self.to_mot = nn.Sequential(nn.LeakyReLU(0.2),
                                    nn.Conv2d(channels, out_channels, (1, 1)))
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
        else:
            self.edge_importance_bt = 1
            self.edge_importance_b = 1
            self.edge_importance_m = 1
            self.edge_importance_j = 1

    def forward(self, x, sty_leftleg, sty_rightleg, sty_spine, sty_leftarm, sty_rightarm):
        r"""
            x: (n, c, t, v)
            sty_features: [(n, 4c, 5, t), ..., (n, c, 21, 4t)] 
        """
        # bottleneck
        x = self.bottleneck(x, 
                            sty_leftleg[0], sty_rightleg[0], 
                            sty_spine[0], 
                            sty_leftarm[0], sty_rightarm[0],
                            self.A_b * self.edge_importance_bt)
        
        # G3
        x = self.bodypart(x, 
                          sty_leftleg[1], sty_rightleg[1], 
                          sty_spine[1], 
                          sty_leftarm[1], sty_rightarm[1], 
                          self.A_b * self.edge_importance_b)
        
        # G2
        x = self.up_BodypartToMid(x)
        x = self.up_temp1(x, scale_factor=(2, 1), mode='nearest')
        x = self.mid(x, 
                     sty_leftleg[2], sty_rightleg[2], 
                     sty_spine[2], 
                     sty_leftarm[2], sty_rightarm[2], 
                     self.A_m * self.edge_importance_m)
        
        # G1
        x = self.up_MidToJoint(x)
        x = self.up_temp2(x, scale_factor=(2, 1), mode='nearest')
        x = self.joint(x, 
                       sty_leftleg[3], sty_rightleg[3], 
                       sty_spine[3], 
                       sty_leftarm[3], sty_rightarm[3], 
                       self.A_j * self.edge_importance_j)

        # to mot
        x = self.to_mot(x)
        
        return x.permute(0, 1, 3, 2).contiguous()   # (N, C, V, T)
    

def mixing_styles(style_cnt, style_cls, prob):
    bdy_idx = [0, 1, 2, 3, 4]
    bdy_part_idx = [[0, 1], [2], [3, 4]]    # leg, spine, arm = [0, 1], [2], [3, 4]
    
    bdy_part_select, bdy_part_not_select = [], []
    if prob > 0 and random.random() < prob:
        n_choice = random.randint(1, 3)
        bdy_part_select = random.sample(bdy_part_idx, n_choice)
        bdy_part_select = sum(bdy_part_select, [])
        bdy_part_select.sort()
        bdy_part_not_select = [x for x in bdy_idx if x not in bdy_part_select]

        if len(bdy_part_select) == 5:
            istyles = [style_cnt]*5
        else:
            istyles = [None]*5
            for i in range(len(bdy_part_select)):
                istyles[bdy_part_select[i]] = style_cnt
            for j in range(len(bdy_part_not_select)):
                istyles[bdy_part_not_select[j]] = style_cls
    else:
        istyles = [style_cls]*5

    return istyles, bdy_part_select


if __name__ == '__main__':
    import argparse
    sys.path.append('./etc')
    from utils import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()
    config = get_config(args.config)

    G = Generator(config['model']['gen'])
    xa = torch.randn(2, 12, 21, 240)
    xb = torch.randn(2, 12, 21, 120)

    xaa, xbb, xab, xaba, xabb  = G(xa, xb)

    print(xab.shape)