import torch 
from torch import nn 

class LogitLossMSE(nn.Module):
    def __init__(self):
        super(LogitLossMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, distilled_img_logits, real_img_logits):
        return self.mse(distilled_img_logits, real_img_logits)
    
class LogitLossKLDiv(nn.Module):
    def __init__(self):
        super(LogitLossKLDiv, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, distilled_img_logits, real_img_logits):
        log_probs = nn.functional.log_softmax(distilled_img_logits, dim=1)
        probs = nn.functional.softmax(real_img_logits, dim=1)
        return self.kl_div(log_probs, probs)

class FeatureLossMSE(nn.Module):
    def __init__(self):
        super(FeatureLossMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, feat_distilled, feat_real):
        return self.mse(feat_distilled, feat_real)