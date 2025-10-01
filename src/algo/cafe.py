from .interface import IDatasetCondensation
from utils import DiffAugment, ParamDiffAug, evaluate_dii_method, get_images

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models import get_model_by_name
import copy

def criterion_middle(real_feature, syn_feature):
    MSE_Loss = nn.MSELoss(reduction='sum')
    shape_real = real_feature.shape
    real_feature = torch.mean(real_feature.view(10, shape_real[0] // 10, *shape_real[1:]), dim=1)

    shape_syn = syn_feature.shape
    syn_feature = torch.mean(syn_feature.view(10, shape_syn[0] // 10, *shape_syn[1:]), dim=1)

    return MSE_Loss(real_feature, syn_feature)

class CAFE(IDatasetCondensation):
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
        self.forth_weight = opt['forth_weight']
        self.third_weight = opt['third_weight']
        self.second_weight = opt['second_weight']
        self.first_weight = opt['first_weight']
        self.inner_weight = opt['inner_weight']

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

    def update_model(optimizer, steps, loss_function, net, syn_data_data, syn_data_target): 
        for s in range(steps):
            net.train()
            prediction_syn = net(syn_data_data)
            loss_syn = loss_function(prediction_syn, syn_data_target)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

    def distillation(self, distillation_steps: int, network_step: int):

        for i in range(1): 
            data_syn = torch.randn(size=(self.n_classes * self.ipc, self.channel, self.img_size[0], self.img_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
            targets_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.n_classes)], dtype=torch.long, requires_grad=False,  device=self.device).view(-1)

            optimizer_img  = torch.optim.SGD([data_syn, ], lr=self.lr_img)
            optimizer_img.zero_grad()

            loss_fn = nn.CrossEntropyLoss().to(self.device)
            criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(self.device)

            for k in tqdm(range(distillation_steps)):
                net = get_model_by_name(self.model_name).to(self.device)
                net.train() 

                net_params = list(net.parameters()) 
                optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)
                optimizer_net.zero_grad()

                loss_avg = 0
                loss_kai = 0
                loss_middle_item = 0

                
                img_real_gather = []
                img_syn_gather = []
                lab_real_gather = []
                lab_syn_gather = []

                loss = torch.tensor(0.0).to(self.device)
                for c in range(self.n_classes): 
                    img_real = get_images(self.indices_class, self.images_all, c, self.batch_size) 
                    img_syn = data_syn[c * self.ipc:(c + 1) * self.ipc].reshape((self.ipc, self.channel, self.img_size[0], self.img_size[1]))
                    
                    if self.dsa_param is not None:
                        img_real = DiffAugment(img_real, strategy=self.augment_strategy, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, strategy=self.augment_strategy, param=self.dsa_param)

                    target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=self.device) * c
                    target_syn = torch.ones((self.ipc,), dtype=torch.long, device=self.device) * c
                        
                    img_real_gather.append(img_real)
                    lab_real_gather.append(target_real)
                    img_syn_gather.append(img_syn)
                    lab_syn_gather.append(target_syn)

                img_real_gather = torch.stack(img_real_gather, dim=0).reshape(self.batch_size * 10, 3, 32, 32)
                img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(self.ipc * 10, 3, 32, 32)
                lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(self.batch_size * 10)
                lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(self.ipc * 10)

                output_real, feature_real = net(img_real_gather, return_feature=True)
                output_syn, feature_syn = net(img_syn_gather, return_feature=True)

                loss_middle = self.forth_weight(criterion_middle(feature_real[-1], feature_syn[-1]) + self.third_weight * criterion_middle(feature_real[-2], feature_syn[-2]) + self.second_weight * criterion_middle(feature_real[-3], feature_syn[-3]) + self.first_weight * criterion_middle(feature_real[-4], feature_syn[-4]))
                loss_real = loss_fn(output_real, lab_real_gather)
                loss += loss_middle + loss_real

                last_real_feature = torch.mean(feature_real[0].view(10, int(feature_real[0].shape[0] / self.n_classes), feature_real[0].shape[1]), dim=1)
                last_syn_feature = torch.mean(feature_syn[0].view(10, int(feature_syn[0].shape[0] / self.n_classes), feature_syn[0].shape[1]), dim=1)
                output = torch.mm(feature_real[0], last_syn_feature.t())

                last_real_feature = torch.mean(
                    last_real_feature.unsqueeze(0).reshape(10, int(last_real_feature.shape[0] / self.n_classes),
                                                  last_real_feature.shape[1]), dim=1
                )
                loss_output = criterion_middle(last_syn_feature, last_real_feature) + self.inner_weight * criterion_sum(output, lab_real_gather)
                loss += loss_output
                
                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()
                loss_avg += loss.item()
                loss_kai += loss_output.item()
                loss_middle_item += loss_middle.item()

                image_syn_train, label_syn_train = copy.deepcopy(data_syn.detach()), copy.deepcopy(
                targets_syn.detach()) 

                self.update_model(optimizer_net, network_step, loss_fn, net, image_syn_train, label_syn_train)
                loss_avg /= (self.n_classes * self.ipc)

                if k % 50 == 0:
                    print(f"Step {k}/{distillation_steps}, Loss: {loss_avg:.4f}")
                
                if k == distillation_steps - 1:
                    model_save_name = f'{self.model_name}_ipc{self.images_allipc}_step{k}.pth'
                    path = f'pretrained_models/dc/{model_save_name}' 
                    torch.save(data_syn, path)
        print("Finished Distillation") 

    def evaluate(self, num_train_epochs: int) -> float:
        return evaluate_dii_method(self.model_name, self.synthetic_datas, self.testloader, self.batch_size, self.ipc, num_train_epochs, self.n_classes, self.device)