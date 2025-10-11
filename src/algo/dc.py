# Dataset Condensation with Gradient Matching 

from .interface import IDatasetCondensation

from ..models import get_model_by_name
from ..utils import gradient_distance, evaluate_dii_method, get_images

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class GradientMatching(IDatasetCondensation):
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

    @staticmethod
    def update_model(optimizer, steps, loss_function, net, syn_data_data, syn_data_target):
        for s in range(steps):
            net.train()
            prediction_syn = net(syn_data_data)
            loss_syn = loss_function(prediction_syn, syn_data_target)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()

    def condensation(self, distillation_steps: int, network_step: int):


        for i in range(1):
            data_syn = torch.randn(size=(self.n_classes * self.ipc, self.channel, self.img_size[0], self.img_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
            targets_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.n_classes)], dtype=torch.long, requires_grad=False,  device=self.device).view(-1)

            optimizer_img  = torch.optim.SGD([data_syn, ], lr=self.lr_img)
            optimizer_img.zero_grad()

            loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

            for k in tqdm(range(distillation_steps)):
                net = get_model_by_name(self.model_name, self.opt).to(self.device)
                net.train()

                net_params = list(net.parameters())
                optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)
                optimizer_net.zero_grad()

                loss_avg = 0

                for t in range(self.ipc):
                    loss = torch.tensor(0.0).to(self.device)

                    for c in range(self.n_classes):
                        img_real = get_images(self.indices_class, self.images_all, c, self.batch_size)
                        target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=self.device) * c
                        prediction_real = net(img_real)
                        loss_real = loss_fn(prediction_real, target_real)
                        gw_real = torch.autograd.grad(loss_real, net_params)

                        img_syn = data_syn[c * self.ipc:(c + 1) * self.ipc].reshape((self.ipc, self.channel, self.img_size[0], self.img_size[1]))
                        target_syn = torch.ones((self.ipc,), dtype=torch.long, device=self.device) * c
                        prediction_syn = net(img_syn)
                        loss_syn = loss_fn(prediction_syn, target_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_params, create_graph=True)

                        dist = gradient_distance(gw_syn, gw_real, self.device)
                        loss += dist

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()

                    if t == self.ipc - 1:
                        break

                    self.update_model(optimizer_net, network_step, loss_fn, net, data_syn, targets_syn)

                loss_avg /= (self.n_classes * self.ipc)
                if (k + 1) % 50 == 0:
                    print(f"Step {k + 1}/{distillation_steps}, Loss: {loss_avg:.4f}")
                    model_save_name = f'{self.model_name}_ipc{self.ipc}_step{k}.pt'
                    path = f'pretrained/dc/{model_save_name}'
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(data_syn, path)

            self.synthetic_datas.append(data_syn)
            print(f"Distillation dataset {i+1}/{1} completed.")

    def evaluate(self, num_train_epochs: int) -> float:
        return evaluate_dii_method(self.model_name, self.opt, self.synthetic_datas, self.testloader, self.batch_size, self.ipc, num_train_epochs, self.n_classes, self.device)
