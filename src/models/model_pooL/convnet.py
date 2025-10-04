import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config


#to calculate image output size
def calculate(size, kernel, stride, padding):
  return int(((size+(2*padding)-kernel)/stride) + 1)

#based on https://cs231n.github.io/convolutional-networks/
class CNN(torch.nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()
        channel = opt['channel']
        im_size = opt['img_size']

        outsize = im_size
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=128, kernel_size=3, padding=1) #32*32
        outsize = calculate(outsize,3,1,1)
        self.norm1 = nn.GroupNorm(128, 128)
        self.avg_pooling1 = nn.AvgPool2d(kernel_size=2, stride=2) # (n+2p-f)/s+1 => 32+0-2/2 + 1 =16
        outsize = calculate(outsize,2,2,0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) #out = (n+2p-f)/s+1 => 16+2-3/1 + 1 => 16
        outsize = calculate(outsize,3,1,1)
        self.norm2 = nn.GroupNorm(128, 128)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=2, stride=2) #out = (n+2p-f)/s+1 => 16+0-2/2 +1 => 8
        outsize = calculate(outsize,2,2,0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) #out = (n+2p-f)/s+1 => 8+2-3/1 + 1 => 8
        outsize = calculate(outsize,3,1,1)
        self.norm3 = nn.GroupNorm(128, 128)
        self.avg_pooling3 = nn.AvgPool2d(kernel_size=2, stride=2) #out = (n+2p-f)/s+1 => 8+0-2/2 +1 => 4
        outsize = calculate(outsize,2,2,0)
        self.classifier = nn.Linear(outsize*outsize*128, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.avg_pooling1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.avg_pooling2(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.avg_pooling3(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def embed(self, x): 
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.avg_pooling1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.avg_pooling2(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.avg_pooling3(out)
        out = out.reshape(out.size(0), -1)
        return out
    
    def get_feature_after_forward(self, x):
        feats = []

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.avg_pooling1(out)
        feats.append(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.avg_pooling2(out)
        feats.append(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.avg_pooling3(out)
        feats.append(out)
        out = out.reshape(out.size(0), -1)

        logits = self.classifier(out)
        return logits, feats