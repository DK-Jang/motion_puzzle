import sys
import torch
from torch import nn
sys.path.append('./net')


class BPStyleNet(nn.Module):
    def __init__(self,
                 style_dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 activation='lrelu'):
        super().__init__()

        assert len(kernel_size) == 2
        # assert kernel_size[0] % 2 == 1

        # BP-AdaIN
        self.adain_leftleg = AdaIN(style_dim, in_channels)
        self.adain_rightleg = AdaIN(style_dim, in_channels)
        self.adain_spine = AdaIN(style_dim, in_channels)
        self.adain_leftarm = AdaIN(style_dim, in_channels)
        self.adain_rightarm = AdaIN(style_dim, in_channels)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        
        # stgcn1
        padding = ((kernel_size[0] - 1) // 2, 0)    # same as same_padding
        self.gcn1 = SpatialConv(in_channels, 
                                in_channels, 
                                kernel_size[1], 
                                t_kernel_size=1)
        self.tcn1 = nn.Conv2d(in_channels,
                              in_channels,
                              (kernel_size[0], 1),
                              (stride, 1),
                              padding,
                              padding_mode='reflect')        
        
        # BP-ATN
        self.astyle_leftleg = adaptive_style(in_channels)
        self.astyle_rightleg = adaptive_style(in_channels)
        self.astyle_spine = adaptive_style(in_channels)
        self.astyle_leftarm = adaptive_style(in_channels)
        self.astyle_rightarm = adaptive_style(in_channels)

        # stgcn2
        self.gcn2 = SpatialConv(in_channels, 
                                out_channels, 
                                kernel_size[1], 
                                t_kernel_size=1)
        self.tcn2 = nn.Conv2d(out_channels,
                              out_channels,
                              (kernel_size[0], 1),
                              (stride, 1),
                              padding,
                              padding_mode='reflect')

    def forward(self, x, 
                      s_leftleg, s_rightleg,    # (n, c, t, 21)
                      s_spine, 
                      s_leftarm, s_rightarm, 
                      A):
        
        if A.shape[-1] == 21:      # G1 level
            idx_leftleg = [1, 2, 3, 4]
            idx_rightleg = [5, 6, 7, 8]
            idx_spine = [0, 9, 10, 11, 12]
            idx_leftarm = [13, 14, 15, 16]
            idx_rightarm = [17, 18, 19, 20]

        elif A.shape[-1] == 10:    # G2 level
            idx_leftleg = [0, 1]
            idx_rightleg = [2, 3]
            idx_spine = [4, 5]
            idx_leftarm = [6, 7]
            idx_rightarm = [8, 9]

        elif A.shape[-1] == 5:     # G3 level
            idx_leftleg = [0]
            idx_rightleg = [1]
            idx_spine = [2]
            idx_leftarm = [3]
            idx_rightarm = [4]

        else:
            assert A.shape[-1] == 21 or 10 or 5, "Graph is wrong!!"

        x_leftleg = x[..., idx_leftleg]
        s_leftleg = s_leftleg[..., idx_leftleg]
        x_rightleg = x[..., idx_rightleg]
        s_rightleg = s_rightleg[..., idx_rightleg]
        x_spine = x[..., idx_spine]
        s_spine = s_spine[..., idx_spine]
        x_leftarm = x[..., idx_leftarm]
        s_leftarm = s_leftarm[..., idx_leftarm]
        x_rightarm = x[..., idx_rightarm]
        s_rightarm = s_rightarm[..., idx_rightarm]
        
        # BP-AdaIN
        x_leftleg = self.adain_leftleg(x_leftleg, s_leftleg)
        x_rightleg = self.adain_rightleg(x_rightleg, s_rightleg)
        x_spine = self.adain_spine(x_spine, s_spine)
        x_leftarm = self.adain_leftarm(x_leftarm, s_leftarm)
        x_rightarm = self.adain_rightarm(x_rightarm, s_rightarm)

        x = torch.cat((x_leftleg, x_rightleg, x_spine, x_leftarm, x_rightarm), -1)
        if A.shape[-1] == 21:
            x = torch.cat((x[..., 8:9], x[..., 0:8], x[..., 9:]), -1)

        if self.activation:
            x = self.activation(x)
        
        x, _ = self.gcn1(x, A)
        x = self.tcn1(x)

        x_leftleg = x[..., idx_leftleg]
        x_rightleg = x[..., idx_rightleg]
        x_spine = x[..., idx_spine]
        x_leftarm = x[..., idx_leftarm]
        x_rightarm = x[..., idx_rightarm]

        # BP-ATN
        x_leftleg = self.astyle_leftleg(x_leftleg, s_leftleg)
        x_rightleg = self.astyle_rightleg(x_rightleg, s_rightleg)
        x_spine = self.astyle_spine(x_spine, s_spine)
        x_leftarm = self.astyle_leftarm(x_leftarm, s_leftarm)
        x_rightarm = self.astyle_rightarm(x_rightarm, s_rightarm)

        x = torch.cat((x_leftleg, x_rightleg, x_spine, x_leftarm, x_rightarm), -1)
        if x.shape[-1] == 21:
            x = torch.cat((x[..., 8:9], x[..., 0:8], x[..., 9:]), -1)

        x, _ = self.gcn2(x, A)
        x = self.tcn2(x)

        return x


class ResBPStyleNet(nn.Module):     # not down or up sampling
    def __init__(self, style_dim, dim_in, dim_out, kernel_size, stride, activation='relu'):
        super(ResBPStyleNet, self).__init__()
        self.res = nn.ModuleList()
        self.res += [BPStyleNet(style_dim,
                                dim_in, dim_in, 
                                kernel_size=kernel_size,     # tuple
                                stride=stride,
                                activation=activation)]
        self.res += [BPStyleNet(style_dim,
                                dim_in, dim_out, 
                                kernel_size=kernel_size,
                                stride=stride,
                                activation='none')]
        
        if (dim_in == dim_out) and (stride == 1):
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(dim_in,
                                      dim_out,
                                      kernel_size=1,
                                      stride=(stride, 1))
                
    def forward(self, x, 
                      s_leftleg, s_rightleg, 
                      s_spine, 
                      s_leftarm, s_rightarm, 
                      A):
        x_org = self.shortcut(x)
        for i, layer in enumerate(self.res):
            x = layer(x, s_leftleg, s_rightleg, s_spine, s_leftarm, s_rightarm, A)
        out = x_org + 0.1 * x
        return out


class ResStgcnBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, norm='in', activation='lrelu'):
        super(ResStgcnBlock, self).__init__()
        self.res = nn.ModuleList()
        self.res += [StgcnBlock(dim_in, dim_in, 
                                kernel_size=kernel_size,     # tuple
                                stride=stride,
                                norm=norm,
                                activation=activation)]
        self.res += [StgcnBlock(dim_in, dim_out, 
                                kernel_size=kernel_size,
                                stride=stride,
                                norm=norm,
                                activation='none')]
        
        if (dim_in == dim_out) and (stride == 1):
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(dim_in,
                                      dim_out,
                                      kernel_size=1,
                                      stride=(stride, 1))
                
    def forward(self, x, A):
        x_org = self.shortcut(x)
        for i, layer in enumerate(self.res):
            x = layer(x, A)
        out = x_org + 0.1 * x
        return out


class StgcnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 norm='none',
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        # assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)    # same as same_padding

        # initialize normalization
        norm_dim = in_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        
        self.gcn = SpatialConv(in_channels, 
                               out_channels, 
                               kernel_size[1], 
                               t_kernel_size=1)
        self.tcn = nn.Conv2d(out_channels,
                             out_channels,
                             (kernel_size[0], 1),
                             (stride, 1),
                             padding,
                             padding_mode='reflect')

    def forward(self, x, A):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        x, A = self.gcn(x, A)
        x = self.tcn(x)

        return x


class SpatialConv(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              padding_mode='reflect',
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


# Attention module
class adaptive_style(nn.Module):
    def __init__(self, in_ch):
        super(adaptive_style, self).__init__()
        self.in_ch = in_ch
        self.f = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.g = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.h = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.k = nn.Conv2d(in_ch, in_ch, (1, 1))
    
    def forward(self, x, s_sty, return_nl_map=False):
        r"""
            x: (n, c, t1, v)
            s_sty: (n, c, t2, v)
        """
        b = s_sty.shape[0]

        F = self.f(nn.functional.instance_norm(x)) 
        G = self.g(nn.functional.instance_norm(s_sty))
        H = self.h(s_sty)

        F = F.view(b, self.in_ch, -1).permute(0, 2, 1)
        G = G.view(b, self.in_ch, -1)
        S = torch.bmm(F, G)

        S = self.sm(S)
        H = H.view(b, self.in_ch, -1)
        O = torch.bmm(H, S.permute(0, 2, 1))

        O = O.view(x.size())

        O = self.k(O)
        O += x
        
        if return_nl_map:
            return O, S
        return O


class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.to_latent = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                       nn.Conv2d(num_features, latent_dim, 1, 1, 0),
                                       nn.LeakyReLU(0.2))
        self.inject = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(latent_dim, num_features*2))
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        s = self.to_latent(s).squeeze(-1).squeeze(-1)
        h = self.inject(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta