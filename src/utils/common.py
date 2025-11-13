import os
import random 
import torch
import numpy as np
import yaml 
from easydict import EasyDict
import matplotlib.pyplot as plt

from src.models import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> EasyDict:
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))    
    return config

def get_configs(algo_name: str) -> EasyDict:
    gan_distillation_config = load_config('./config/gan_distillation_config.yaml')
    normal_distillation_config = load_config('./config/normal_distillation_config.yaml')
    distill_algo = algo_name.split('_')[0]
    if distill_algo in ['dim', 'dcm']: 
        return gan_distillation_config[algo_name]
    
    return normal_distillation_config[algo_name]

def get_model_by_name(model_name, opt): 

    if model_name.lower() == 'cnn':
        model = CNN(opt)
    elif model_name.lower() == 'mlp': 
        model = MLP(channel=opt['channel'], im_size=opt['im_size'], num_classes=opt['n_classes'])
    elif model_name.lower() == 'lenet':
        model = LeNet(channel=opt['channel'], num_classes=opt['n_classes'])
    elif model_name.lower() == 'alexnet':
        model = AlexNet(channel=opt['channel'], num_classes=opt['n_classes'], img_size=opt['img_size'])
    else: 
        raise ValueError(f"Model {model_name} not recognized. Available models: cnn, mlp, lenet, alexnet.")

    return model

def showImage(images, save_=False, algo_name=None):
    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images,axes = (1,2,0)))
    plt.axis('off')
    if save_ and algo_name is not None:
        save_dir = "./images/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + f"{algo_name}_sample_image.png")

def get_random_model_from_model_pool(opt):
    idx = random.randint(1, 4)
    dataset_name = opt['dataset_name'].lower()
    model_path = 'pretrained/model_pool/'
    if idx == 1:
        model = CNN(opt)
        model_path = os.path.join(model_path, f'cnn_{dataset_name}.pth')
    elif idx == 2:
        model = AlexNet(channel=opt['channel'], num_classes=opt['n_classes'], img_size=opt['img_size'])
        model_path = os.path.join(model_path, f'alexnet_{dataset_name}.pth')
    elif idx == 3:
        model = LeNet(channel=opt['channel'], num_classes=opt['n_classes'])
        model_path = os.path.join(model_path, f'lenet_{dataset_name}.pth')
    elif idx == 4:
        model = MLP(channel=opt['channel'], im_size=opt['img_size'], num_classes=opt['n_classes'])
        model_path = os.path.join(model_path, f'mlp_{dataset_name}.pth')
    
    model.load_state_dict(torch.load(model_path))
    return model

def get_images(indices_class, images_all, c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

def get_loops(ipc):
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop