import torch 
from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, channel, num_classes, img_size):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channel, img_size, img_size)
            dummy_out = self.features(dummy)
            self._flattened_size = dummy_out.numel()

        self.fc = nn.Linear(self._flattened_size, num_classes) # This will be updated in the first forward pass

    def forward(self, x, return_feature=False):
        if not return_feature:
            x = self.embed(x)
            x = self.fc(x)
            return x
        logits, feat = self.get_features_after_forward(x)
        return logits, feat
    
    def embed(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x

    def get_features_after_forward(self, x):
        features = []
        x = self.features(x)
        features.append(x) 
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x) 

        return logits, features