import torch
import torch.nn as nn
import torchvision.models as models

class SimCLRv2(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLRv2, self).__init__()
        self.backbone = base_model
        num_ftrs = self.backbone.fc.in_features  # Access in_features before replacing
        self.backbone.fc = nn.Identity()  # Remove the original fully connected layer
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections