import torch 
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module): 

    def __init__(self, channel, im_size, num_classes): 

        super(MLP, self).__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channel * im_size * im_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, return_feature=False): 
        if not return_feature:
            x = self.embed(x)
            x = self.fc3(x)
            return x
        logits, feat = self.get_features_after_forward(x)
        return logits, feat

    def embed(self, x): 
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    def get_features_after_forward(self, x):
        features = []

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        features.append(x)

        x = F.relu(self.fc2(x))
        features.append(x)

        logits = self.fc3(x)
        
        return logits, features