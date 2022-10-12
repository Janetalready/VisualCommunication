import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import pdb
import copy

class ReceiverOnestep(nn.Module):
    def __init__(self, device, game_size, width, opt, eps=1e-8):
        super(ReceiverOnestep, self).__init__()
        self.feature = models.vgg16(pretrained=True)

        for child in self.feature.children():
            for param in child.parameters():
                param.requires_grad = False

        num_ftrs = self.feature.classifier[-1].in_features
        self.feature.classifier[-1] = nn.Linear(num_ftrs, 30)
        self.width = 128

    def forward(self, x):
        x = x.view(-1, 3, self.width, self.width)
        probs = self.feature(x/255)
        features = self.feature.features(x/255)
        features = self.feature.avgpool(features)
        features = features.view(-1, 512*7*7)
        features = self.feature.classifier[:-1](features)
        return probs, features

