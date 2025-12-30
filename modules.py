import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel) if filter else nn.Identity(),
            LSFM(out_channel) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.proj_act = nn.GELU()

    def forward(self, x):
        return self.proj_act(self.main(x)) + x

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))

            dynas.append(MPDAF(k, k_out))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes) - 1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class MPDAF(nn.Module):
    def __init__(self, k, k_out):
        super(MPDAF, self).__init__()
        self.k = k
        self.k_out = k_out

        self.conv_l1 = BasicConv(k, k_out, kernel_size=3, stride=1, relu=True)
        self.conv_l2 = nn.Conv2d(k_out, k_out, kernel_size=1)
        self.act1 = nn.Sigmoid()

        self.conv_r1 = BasicConv(k, k_out, kernel_size=3, stride=1, relu=True)
        self.conv_r2 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.act2 = nn.Sigmoid()

        if k != k_out:
            self.downsample = nn.Conv2d(k, k_out, kernel_size=1, bias=False)
        else:
            self.downsample = nn.Identity()

        self.final_conv = nn.Sequential(
            nn.Conv2d(k_out, k_out, kernel_size=1),
            nn.BatchNorm2d(k_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = self.downsample(x)

        xl = self.conv_l1(x)
        xl = F.adaptive_avg_pool2d(xl, 1)
        xl = self.conv_l2(xl)
        xl = self.act1(xl)
        xl = identity * xl

        xr = self.conv_r1(x)
        xr_mean = torch.mean(xr, dim=1, keepdim=True)
        xr_max = torch.max(xr, dim=1, keepdim=True)[0]
        xr = torch.cat([xr_mean, xr_max], dim=1)
        xr = self.conv_r2(xr)
        xr = self.act2(xr)
        xr = identity * xr

        out = identity + xl + xr
        out = self.final_conv(out)
        return out

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        ffted = torch.fft.rfft2(x, norm='ortho')
        real = torch.unsqueeze(torch.real(ffted), dim=-1)
        imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat([real, imag], dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, h, w // 2 + 1)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view(batch, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output

class LSFM(nn.Module):
    def __init__(self, dim, kernels=[3, 5, 7]):
        super(LSFM, self).__init__()
        self.dim = dim
        self.num_branches = len(kernels)

        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1)
            )
            self.branches.append(branch)

        total_dim = dim * self.num_branches

        self.FFC = FourierUnit(in_channels=total_dim, out_channels=total_dim)

        self.out_conv = nn.Conv2d(total_dim, dim, kernel_size=1)

        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            out_i = branch(x)
            outs.append(out_i)

        x_cat = torch.cat(outs, dim=1)

        x_fft = self.FFC(x_cat) + x_cat

        x_out = self.out_conv(x_fft)

        return x_out
