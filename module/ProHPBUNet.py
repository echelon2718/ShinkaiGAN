import torch
import torch.nn as nn
import torch.nn.functional as F

from module.cbam import CBAM
from module.hybrid_perception_module import HPB

class InjectNoise(nn.Module):
    def __init__(self, channels):
        super(InjectNoise, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(channels)[None, :, None, None]
        )
    
    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device)
        return image + self.weight * noise

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, channels, w_dim, hidden_dim=2048):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        ##### NEW, AUTOENCODER #####
        self.style_scale_receptor = nn.Linear(w_dim, hidden_dim)
        self.style_shift_receptor = nn.Linear(w_dim, hidden_dim)
        
        self.style_scale_transform = nn.Linear(hidden_dim, channels)
        self.style_shift_transform = nn.Linear(hidden_dim, channels)
        
    def forward(self, img, w):
        b,c,height,width = w.shape
        w = w.view(b, c*height*width)
        # w = w.flatten().unsqueeze(0)
        # print("UKURAN W", w.shape)
        normalized_img = self.instance_norm(img)
        style_scale = self.style_scale_transform(
            F.relu(self.style_scale_receptor(w))
        )[:, :, None, None]
        
        style_shift = self.style_shift_transform(
            F.relu(self.style_shift_receptor(w))
        )[:, :, None, None]
        # print("SECTION ADAIN")
        # print(style_scale.shape)
        # print(normalized_img.shape)
        transformed_img = style_scale * normalized_img + style_shift
        return transformed_img

class F_RGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(F_RGB, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, img):
        return self.conv(img)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=0.2, downsample=False):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(n_channels=out_channels, reduction_ratio=2, kernel_size=3)
        self.cbam2 = CBAM(n_channels=out_channels, reduction_ratio=2, kernel_size=3)
        self.instancenorm2d =  nn.InstanceNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(leakiness, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)
        x = self.instancenorm2d(x)
        x = self.leakyrelu(x)
        
        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.instancenorm2d(x)
        x = self.leakyrelu(x)
        
        if self.downsample:
            skip_x = x
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
            return x, skip_x
        return x, None

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=8192, leakiness=0.2, upsample=False):
        super(UpBlock, self).__init__()
        self.begin_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.leakyrelu = nn.LeakyReLU(leakiness, inplace=True)
        self.adain1 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.adain2 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.upsample = upsample

    def forward(self, x_g, x_s, att_vec):
        x = x_g
        if x_s is not None:
            x = torch.cat([x_g, x_s], dim=1)
            x = self.begin_conv(x)
        
        x = self.conv1(x)
        x = self.inject_noise1(x)
        x = self.leakyrelu(x)
        x = self.adain1(x, att_vec)

        # x = self.conv2(x)
        x = self.inject_noise2(x)
        x = self.leakyrelu(x)
        x = self.adain2(x, att_vec)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            
        return x
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.hpb1 = HPB(out_channels, attn_height_top_k=out_channels, attn_width_top_k=out_channels)
        self.hpb2 = HPB(out_channels, attn_height_top_k=out_channels, attn_width_top_k=out_channels)
        self.hpb3 = HPB(out_channels, attn_height_top_k=out_channels, attn_width_top_k=out_channels)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv_in(x)
        x, _ = self.hpb1(x)
        x, _ = self.hpb2(x)
        x, attn_vec = self.hpb3(x)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.upsample(x)

        return x, attn_vec

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down_levels = nn.ModuleList([
            #### LEVEL 4 #### (16, 512, 512) --> (64, 256, 256) x2 down, skips
            DownBlock(16, 16),   #15 #XXX
            DownBlock(16, 32),   #14
            DownBlock(32, 32),   #13
            DownBlock(32, 32),   #12
            DownBlock(32, 64),   #11
            DownBlock(64, 64),   #10
            DownBlock(64, 64, downsample=True),   #9

            #### LEVEL 3 #### (64, 256, 256) --> (128, 128, 128)
            DownBlock(64, 128),  #8 #XXX
            DownBlock(128, 128), #7       
            DownBlock(128, 128, downsample=True), #6

            #### LEVEL 2 #### (128, 128, 128) --> (256, 64, 64)
            DownBlock(128, 256), #5  #XXX
            DownBlock(256, 256), #4       
            DownBlock(256, 256, downsample=True), #3     

            #### LEVEL 1 #### (256, 64, 64) --> (512, 64, 64)
            DownBlock(256, 512), #2  #XXX 
            DownBlock(512, 512), #1     

            #### LEVEL 0 IN #### (512, 64, 64) --> (1024, 32, 32)
            DownBlock(512, 512, downsample=True), #0 --> Nyambung ke bottleneck           
        ])

        self.up_levels = nn.ModuleList([
            #### LEVEL 0 OUT #### (1024, 32, 32) --> (512, 64, 64)
            BottleNeck(512, 512), #0 --> Keluar bottleneck, upsample 2x
            
            #### LEVEL 1 #### (512, 64, 64) --> (1024, 64, 64) --> (256, 128, 128)
            UpBlock(512, 512), #1 
            UpBlock(512, 512, upsample=True), #2 #XXX

            #### LEVEL 2 #### (256, 128, 128) --> (512, 128, 128) --> (128, 256, 256)
            UpBlock(512, 256), #3 
            UpBlock(256, 256), #4
            UpBlock(256, 256, upsample=True), #5 #XXX

            #### LEVEL 3 #### (128, 256, 256) --> (256, 256, 256) --> (64, 512, 512)
            UpBlock(256, 128), #6 #XXX
            UpBlock(128, 128), #7
            UpBlock(128, 128, upsample=True), #8 #XXX

            #### LEVEL 4 #### (64, 512, 512) --> (128, 512, 512) --> (16, 512, 512)
            UpBlock(128, 64), #9 #XXX
            UpBlock(64, 64), #10
            UpBlock(64, 32), #11
            UpBlock(32, 32), #12
            UpBlock(32, 32), #13
            UpBlock(32, 16), #14
            UpBlock(16, 16), #15
        ])

        self.from_rgb = nn.ModuleList([
            F_RGB(3, 16),
            F_RGB(3, 16),
            
            F_RGB(3, 16),
            F_RGB(3, 32),
            F_RGB(3, 32),
            
            F_RGB(3, 32),
            F_RGB(3, 64),
            F_RGB(3, 64),
            
            F_RGB(3, 64),
            F_RGB(3, 128),
            F_RGB(3, 128),
            
            F_RGB(3, 128),
            F_RGB(3, 256),
            F_RGB(3, 256),
            
            F_RGB(3, 256), #Level 2
            F_RGB(3, 512), #Level 1
        ])

        self.to_rgb = nn.ModuleList([
            F_RGB(16, 3),
            F_RGB(16, 3),
            
            F_RGB(32, 3),
            F_RGB(32, 3),
            F_RGB(32, 3),
            
            F_RGB(64, 3),
            F_RGB(64, 3),
            F_RGB(64, 3),
            
            F_RGB(128, 3),
            F_RGB(128, 3),
            F_RGB(128, 3),
            
            F_RGB(256, 3),
            F_RGB(256, 3),
            F_RGB(256, 3),
            
            F_RGB(512, 3),
            F_RGB(512, 3),
        ])

    def forward(self, input_img, depth):
        x = self.from_rgb[-depth](input_img)
        skips = []
        
        for i in range(15 - depth, 16):
            x, skip = self.down_levels[i](x)
            skips.append(skip)

        x, attn_vec = self.up_levels[0](x)
        
        for i in range(1, depth + 1):
            skip = skips.pop()
            x = self.up_levels[i](x, skip, attn_vec)

        x = self.to_rgb[-depth](x)

        return x

class DiscDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(DiscDownBlock, self).__init__()
        self.layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='reflect'),
                        nn.InstanceNorm2d(out_channels),
                        nn.LeakyReLU(0.2)
                    )

    def forward(self, x):
        x = self.layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down_levels = nn.ModuleList([
            DiscDownBlock(16, 64),
            DiscDownBlock(64, 128),
            DiscDownBlock(128, 256),
            DiscDownBlock(256, 512),        
        ])

        self.final_conv = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.from_rgb = nn.ModuleList([
            F_RGB(3, 16),
            F_RGB(3, 64),
            F_RGB(3, 128),
            F_RGB(3, 256),
        ])

    def forward(self, input_img, depth, return_features=False):
        depth_to_index = {2: 3, 5: 2, 8: 1, 15: 0}
        index_to_depth = {0: 15, 1: 8, 2: 5, 3: 2}
        
        features = []
        
        index = depth_to_index[depth]
        
        x = self.from_rgb[index](input_img)
        
        features.append(x)

        for i in range(index, 4):
            x = self.down_levels[i](x)
            features.append(x)
            
        x = self.final_conv(x)
        x = self.sigmoid(x)

        if return_features:
            return x, features
        return x