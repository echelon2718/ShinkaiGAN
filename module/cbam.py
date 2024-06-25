import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, n_channels, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels = n_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(self.n_channels, self.reduction_ratio)
        self.spatial_attention = SpatialAttention(self.kernel_size)
    
    def forward(self, x):
        ca = self.channel_attention(x)
        fp = ca * x
        sa = self.spatial_attention(fp)
        fpp = sa * fp
        return fpp

class ChannelAttention(nn.Module):
    def __init__(self, n_channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels = n_channels
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = self.n_channels // self.reduction_ratio
        self.fc = nn.Sequential(
            nn.Linear(self.n_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_channels)
        )
    
    def forward(self, x):
        kernel = (x.size(-2), x.size(-1)) # input shape: (batch_size, n_channels, height, width)
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)

        avg_pool = self.fc(avg_pool)
        max_pool = self.fc(max_pool)

        pool_sum = avg_pool + max_pool

        refined_features_ca = torch.sigmoid(pool_sum).unsqueeze(2).unsqueeze(3)
        refined_features_ca = refined_features_ca.repeat(1, 1, kernel[0], kernel[1])
        return refined_features_ca

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=int((kernel_size-1)/2))
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = self.batchnorm(conv)
        
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x