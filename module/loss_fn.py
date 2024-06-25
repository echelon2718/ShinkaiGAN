import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

class PerceptualLoss(nn.Module):
    '''
    This class is used to calculate the perceptual loss. It uses a pre-trained VGG16 model to calculate the loss.
    '''
    def __init__(self, vgg_model, type_loss="custom"):
        super(PerceptualLoss, self).__init__()
        self.type_loss = type_loss
        self.vgg_model = vgg_model
        self.vgg_layers = self.vgg_model.features
        if type_loss == "content":
            self.layer_names = ['4', '9', '18', '27', '36']
        elif type_loss == "style":
            self.layer_names = ['0', '5', '10', '19', '28']
        else:
            self.layer_names = ['0', '1', '2', '3', '4', '5']
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        x_features = self.get_features(x)
        y_features = self.get_features(y)

        loss = 0

        if self.type_loss=="custom":
            i = 0
            weight = [0.88, 0.79, 0.63, 0.51, 0.39, 1.07]
            
            for x_feature, y_feature in zip(x_features, y_features):
                loss += weight[i] * self.loss(x_feature, y_feature)
                i += 1

            return loss
        
        for x_feature, y_feature in zip(x_features, y_features):
            loss += self.loss(x_feature, y_feature)

        return loss

    def get_features(self, x):
        features = []
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features.append(x)

        return features

# class ContrastiveLoss(nn.Module):
#     def __init__(self, model, temperature=0.5, device="cuda"):
#         super(ContrastiveLoss, self).__init__()
#         self.model = model.to(device)
#         self.temperature = temperature
#         self.device = device

#     def forward(self, X, Y):
#         # Move inputs to the appropriate device
#         X, Y = X.to(self.device), Y.to(self.device)

#         with torch.no_grad():
#             # Pass through the model to get the features
#             generated_features = F.normalize(self.model(X), dim=-1)
#             style_features = F.normalize(self.model(Y), dim=-1)
        
#         # Compute similarity matrix
#         similarity_matrix = torch.matmul(generated_features, style_features.T)
        
#         # Compute the contrastive loss (NT-Xent)
#         labels = torch.arange(similarity_matrix.size(0)).to(self.device)
#         logits = similarity_matrix / self.temperature
#         loss = F.cross_entropy(logits, labels)
        
#         return loss

class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.transform = transforms.Grayscale()

    def forward(self, src, target):
        input_gray = self.transform(src)
        target_gray = self.transform(target)
        return self.mse_loss(input_gray, target_gray).item()
        
class AdversarialLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(AdversarialLoss, self).__init__()
        self.device = device
        self.loss = nn.BCELoss()

    def forward(self, real_X, level, gen, disc):
        real_X = real_X.to(self.device)
        fake_Y = gen(real_X, level)
        preds = disc(fake_Y, level)
        loss = self.loss(preds, torch.ones_like(preds))
        return loss, fake_Y

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, real_features, fake_features):
        loss = 0
        for real, fake in zip(real_features, fake_features):
            loss += self.criterion(fake, real.detach())
        return loss

class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.transform = transforms.Grayscale()

    def forward(self, src, target):
        input_gray = self.transform(src)
        target_gray = self.transform(target)
        return self.mse_loss(input_gray, target_gray)

class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * channels * height * width)

class GaussianBlur(nn.Module):
    def __init__(self, nc, kernlen=21, nsig=3):
        super(GaussianBlur, self).__init__()
        self.nc = nc
        self.kernlen = kernlen
        self.nsig = nsig
        self.weight = self.create_gaussian_kernel()

    def create_gaussian_kernel(self):
        interval = (2 * self.nsig + 1.) / (self.kernlen)
        x = np.linspace(-self.nsig - interval / 2., self.nsig + interval / 2., self.kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((self.kernlen, self.kernlen, 1, 1))
        out_filter = np.repeat(out_filter, self.nc, axis=2)
        out_filter = torch.from_numpy(out_filter).permute(2, 3, 0, 1)
        return nn.Parameter(data=out_filter, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                f"The channel of input [{x.size(1)}] does not match the preset channel [{self.nc}]")
        x = F.conv2d(x, self.weight, stride=1, padding=self.kernlen // 2, groups=self.nc)
        return x

class ColorLoss(nn.Module):
    def __init__(self, nc=3):
        super(ColorLoss, self).__init__()
        self.blur = GaussianBlur(nc)

    def forward(self, x1, x2):
        blur_x1 = self.blur(x1)
        blur_x2 = self.blur(x2)
        return torch.sum(torch.pow((blur_x1 - blur_x2), 2)).div(2 * x1.size(0))

class GeneratorLoss(nn.Module):
    def __init__(self, 
                 adversarial_loss,
                 content_loss,
                 upper_loss,
                 style_loss,
                 color_loss,
                 grayscale_loss,
                 total_variation_loss,
                 feat_match_loss, 
                 
                 lambda_adv=0.1, 
                 lambda_ct=0.1, 
                 lambda_up=0.1, 
                 lambda_style=0.1, 
                 lambda_color=0.01,
                 lambda_grayscale=0.1,
                 lambda_tv=0.1,
                 lambda_fml=0.1,
        ):
        super(GeneratorLoss, self).__init__()
        self.adversarial_loss = adversarial_loss
        self.content_loss = content_loss
        self.upper_loss = upper_loss
        self.style_loss = style_loss
        self.color_loss = color_loss
        self.grayscale_loss = grayscale_loss
        self.total_variation_loss = total_variation_loss
        self.feat_match_loss = feat_match_loss

        self.lambda_adv = lambda_adv
        self.lambda_ct = lambda_ct
        self.lambda_up = lambda_up
        self.lambda_style = lambda_style
        self.lambda_color = lambda_color
        self.lambda_grayscale = lambda_grayscale
        self.lambda_tv = lambda_tv
        self.lambda_fml = lambda_fml

    def forward(self, X, Y, level, gen, disc):
        adv_loss, fake_Y = self.adversarial_loss(X, level, gen, disc)
        ct_loss = self.content_loss(Y, fake_Y)
        up_loss = self.upper_loss(Y, fake_Y)
        style_loss = self.style_loss(Y, fake_Y)
        color_loss = self.color_loss(Y, fake_Y)
        grayscale_loss = self.grayscale_loss(Y, fake_Y)
        tv_loss = self.total_variation_loss(fake_Y)

        with torch.no_grad():
            _, real_features = disc(Y, level, return_features=True)
            _, fake_features = disc(fake_Y, level, return_features=True)

        fml_loss = self.feat_match_loss(real_features, fake_features)

        loss = (
            self.lambda_adv * adv_loss +
            self.lambda_ct * ct_loss +
            self.lambda_up * up_loss +
            self.lambda_style * style_loss +
            self.lambda_color * color_loss +
            self.lambda_grayscale * grayscale_loss +
            self.lambda_tv * tv_loss +
            self.lambda_fml * fml_loss
        )

        return loss, fake_Y

class DiscriminatorLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(DiscriminatorLoss, self).__init__()
        self.device = device
        self.loss = nn.BCELoss()

    def forward(self, real_Y, fake_Y, level, disc):
        real_Y, fake_Y = real_Y.to(self.device), fake_Y.to(self.device)
        preds_real = disc(real_Y, level)
        loss_real = self.loss(preds_real, torch.ones_like(preds_real))  # Target is 1 for real

        preds_fake = disc(fake_Y, level)
        loss_fake = self.loss(preds_fake, torch.zeros_like(preds_fake))  # Target is 0 for fake

        return loss_real + loss_fake