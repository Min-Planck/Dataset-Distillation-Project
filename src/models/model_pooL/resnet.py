import torch 
from torch import nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, opt):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        self.resnet.conv1 = nn.Conv2d(opt['num_channel'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, opt['num_classes'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
     
        logits, _ = self.get_features_after_forward(x)
        return logits

    def get_features_after_forward(self, x: torch.Tensor):
        features = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        features.append(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.resnet.fc(x)

        return logits, features

class ResNet34(nn.Module): 

    def __init__(self, opt):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(weights='IMAGENET1K_V1')

        self.resnet.conv1 = nn.Conv2d(opt['num_channel'], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, opt['num_classes'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.get_features_after_forward(x)
        return logits

    def get_features_after_forward(self, x: torch.Tensor):
 
        features = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        features.append(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.resnet.fc(x)

        return logits, features