from src.utils import IDatasetCondensation, get_model_by_name, Synthetic, get_images, evaluate_dii_method, DiffAugment, ParamDiffAug

import os 
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import copy
import torch.nn as nn

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
        self.lambda_1 = opt['lambda_1']
        self.lambda_2 = opt['lambda_2']

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
        else:
            self.dsa = False

    @staticmethod
    def update_model(optimizer, steps, loss_function, net, syn_data_data, syn_data_target):
        acc_avg = 0.0
        for s in range(steps):
            net.train()
            prediction_syn = net(syn_data_data)
            acc = (prediction_syn.argmax(dim=1) == syn_data_target).sum().item()
            loss_syn = loss_function(prediction_syn, syn_data_target)

            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()
            acc_avg += acc

        return acc_avg / steps

    def condensation(self, distillation_steps: int, outer_loop: int | None, network_step: int | None):

        start_time = time.time()
        for i in range(1):
            data_syn = torch.randn(size=(self.n_classes * self.ipc, self.channel, self.img_size[0], self.img_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
            targets_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.n_classes)], dtype=torch.long, requires_grad=False,  device=self.device).view(-1)

            optimizer_img  = torch.optim.SGD([data_syn, ], lr=self.lr_img)
            optimizer_img.zero_grad()

            loss_fn = nn.CrossEntropyLoss().to(self.device)
            criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(self.device)

            net = get_model_by_name(self.model_name, self.opt).to(self.device)
            net.train()

            outer_acc_watcher = []
            innter_acc_wathcher = []

            inner_loop_cnt = 0
            outer_loop_cnt = 0

            while True:
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

                    if self.dsa:
                        img_real = DiffAugment(img_real, strategy=self.augment_strategy, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, strategy=self.augment_strategy, param=self.dsa_param)

                    target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=self.device) * c
                    target_syn = torch.ones((self.ipc,), dtype=torch.long, device=self.device) * c

                    img_real_gather.append(img_real)
                    lab_real_gather.append(target_real)
                    img_syn_gather.append(img_syn)
                    lab_syn_gather.append(target_syn)

                img_real_gather = torch.stack(img_real_gather, dim=0).reshape(self.batch_size * 10, self.channel, self.img_size[0], self.img_size[1])
                img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(self.ipc * 10, self.channel, self.img_size[0], self.img_size[1])
                lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(self.batch_size * 10)
                lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(self.ipc * 10)

                output_real, feature_real = net(img_real_gather, return_feature=True)
                output_syn, feature_syn = net(img_syn_gather, return_feature=True)

                loss_middle = self.forth_weight * criterion_middle(feature_real[-1], feature_syn[-1]) + self.third_weight * criterion_middle(feature_real[-2], feature_syn[-2]) + self.second_weight * criterion_middle(feature_real[-3], feature_syn[-3]) + self.first_weight * criterion_middle(feature_real[-4], feature_syn[-4])
                loss_real = loss_fn(output_real, lab_real_gather)
                loss += loss_middle + loss_real

                last_real_feature = torch.mean(feature_real[0].view(10, int(feature_real[0].shape[0] / self.n_classes), feature_real[0].shape[1]), dim=1)
                last_syn_feature = torch.mean(feature_syn[0].view(10, int(feature_syn[0].shape[0] / self.n_classes), feature_syn[0].shape[1]), dim=1)
                output = torch.mm(feature_real[0], last_syn_feature.t())

                loss_output = criterion_middle(last_syn_feature, last_real_feature) + self.inner_weight * criterion_sum(output, lab_real_gather)
                loss += loss_output

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()
                loss_avg += loss.item()
                loss_kai += loss_output.item()
                loss_middle_item += loss_middle.item()

                outer_acc = 0
                for c in range(self.n_classes):
                    img_real = get_images(self.indices_class, self.images_all, c, self.batch_size)
                    target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=self.device) * c

                    output = net(img_real)
                    outer_acc += (output.argmax(dim=1) == target_real).sum().item()

                outer_acc /= self.n_classes
                outer_acc_watcher.append(outer_acc)
                outer_loop_cnt += 1

                if len(outer_acc_watcher) == 10:
                    if max(outer_acc_watcher) - min(outer_acc_watcher) < self.lambda_1:
                        outer_acc_watcher = list()
                        outer_loop_cnt = 0
                        outer_acc = 0.0
                        break

                    else:
                        outer_acc_watcher.pop(0)

                image_syn_train, label_syn_train = copy.deepcopy(data_syn.detach()), copy.deepcopy(targets_syn.detach())

                dst_syn = Synthetic(image_syn_train, label_syn_train)
                loader_syn = DataLoader(dst_syn, batch_size=self.batch_size, shuffle=True)

                inner_acc_watcher = list()
                acc_syn_innter_watcher = list()
                inner_cnt = 0
                acc_test = 0

                while True:
                    acc_syn = self.update_model(optimizer_net, network_step, loss_fn, net, image_syn_train, label_syn_train)
                    acc_syn_innter_watcher.append(acc_syn)

                    for c in range(self.n_classes):
                        img_real = get_images(self.indices_class, self.images_all, c, self.batch_size)
                        target_real = torch.ones((img_real.shape[0],), dtype=torch.long, device=self.device) * c

                        output = net(img_real)
                        acc_test += (target_real == output.argmax(dim=1)).sum().item() / self.batch_size

                    acc_test /= self.n_classes
                    inner_acc_watcher.append(acc_test)

                    inner_cnt += 1
                    if len(inner_acc_watcher) == 10:
                        if max(inner_acc_watcher) - min(inner_acc_watcher) < self.lambda_2:
                            inner_acc_watcher = list()
                            inner_cnt = 0
                            acc_test = 0
                            break
                        else:
                            inner_acc_watcher.pop(0)
                loss_avg /= (self.n_classes * self.ipc)
                if (outer_loop_cnt + 1) % 50 == 0:
                    print(f"Step {outer_loop_cnt}, Loss: {loss_avg:.4f}, Loss_middle: {loss_middle_item:.4f}")
                    model_save_name = f'{self.model_name}_ipc{self.ipc}_step{outer_loop_cnt}.pt'
                    path = f'pretrained_models/cafe/{model_save_name}'
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(data_syn, path)

                if outer_loop_cnt == distillation_steps - 1:
                    self.synthetic_datas.append(data_syn)
                    break
        end_time = time.time()
        print("Finished Distillation")

        return end_time - start_time
    def evaluate(self, num_train_epochs: int) -> float:
        return evaluate_dii_method(self.model_name, self.opt, self.synthetic_datas, self.testloader, self.batch_size, self.ipc, num_train_epochs, self.n_classes, self.device)