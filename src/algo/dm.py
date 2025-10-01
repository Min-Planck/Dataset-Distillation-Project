import time
import numpy as np
import torch
import torch.nn as nn
from .interface import IDatasetCondensation
from tqdm import tqdm
from utils import DiffAugment, ParamDiffAug, get_model_by_name, evaluate_dii_method, get_images 

from models import get_ac_gan


class DistributionMatching(IDatasetCondensation):
    

    def __init__(self,
                 model_name,
                 trainset, 
                 testloader,
                 device,
                 ipc,
                 opt):
        
        self.model_name = model_name 
        self.trainset = trainset
        self.testloader = testloader 
        self.opt = opt
        self.device = device
        self.ipc = ipc

        self.n_classes = opt['n_classes']
        self.channel = opt['channel']
        self.batch_size = opt['batch_size'] 
        self.img_size = (opt['img_size'], opt['img_size'])
        self.lr_img = opt['lr_img']
        self.lr_net = opt['lr_net']
        
        self.indices_class = [[] for c in range(self.n_classes)]
        self.images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
        self.labels_all = [trainset[i][1] for i in range(len(trainset))]
        self.synthetic_datas = []

        for i, lab in enumerate(self.labels_all):
            self.indices_class[lab].append(i)

        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)
        if opt['dsa']:
            self.dsa = True
            self.dsa_param = ParamDiffAug()
            self.augment_strategy = opt['dsa_strategy']
            
    def distillation(self, distillation_steps: int, network_step: int):
        for i in range(1): 
            data_syn = torch.randn(size=(self.n_classes * self.ipc, self.channel, self.img_size[0], self.img_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
            targets_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.n_classes)], dtype=torch.long, requires_grad=False,  device=self.device).view(-1)

            optimizer_img  = torch.optim.SGD([data_syn, ], lr=self.lr_img)
            optimizer_img.zero_grad()

            loss_avg = 0 
            loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
            
            loss = torch.tensor(0.0).to(self.device)
            for k in tqdm(range(distillation_steps)):
                net = get_model_by_name(self.model_name, self.opt).to(self.device)
                net.train()

                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed
                image_real_all, image_syn_all = [], []

                for c in range(self.n_classes):
                    img_real = get_images(self.indices_class, self.images_all, c, self.batch_size)
                    img_syn = data_syn[c * self.ipc:(c + 1) * self.ipc].reshape((self.ipc, self.channel, self.img_size[0], self.img_size[1]))

                    if self.dsa:
                        img_real = DiffAugment(img_real, strategy=self.augment_strategy, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, strategy=self.augment_strategy, param=self.dsa_param)

                    image_real_all.append(img_real)
                    image_syn_all.append(img_syn)

                image_real_all = torch.cat(image_real_all, dim=0)
                image_syn_all = torch.cat(image_syn_all, dim=0)

                output_real = embed(image_real_all).detach()
                output_syn = embed(image_syn_all)

                loss += torch.sum((torch.mean(output_real.reshape(self.n_classes, self.batch_size, -1), dim=1) - torch.mean(output_syn.reshape(self.n_classes, self.ipc, -1), dim=1))**2)

                optimizer_img.zero_grad() 
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                loss_avg /= self.n_classes
                if (k+1) % 100 == 0:
                    print(f"Step {k}/{distillation_steps}, Loss: {loss_avg:.4f}")
                    model_save_name = f'{self.model_name}_ipc{self.images_allipc}_step{k}.pth'
                    path = f'pretrained_models/dm/{model_save_name}' 
                    torch.save(data_syn, path)

    def evaluate(self, num_train_epochs: int) -> float:
        return evaluate_dii_method(self.model_name, self.synthetic_datas, self.testloader, self.batch_size, self.ipc, num_train_epochs, self.n_classes, self.device)

    def _get_time(self): 
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())