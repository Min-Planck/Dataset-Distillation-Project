from torch.utils.data import Dataset

class Synthetic(Dataset):
    def __init__(self, data, targets):
        self.data = data.detach().float()
        self.targets = targets.detach()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]