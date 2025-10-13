import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src import DiM, D2M, GradientMatching, DistributionMatching, DSA, CAFE, load_data
from src.models import get_random_model_from_model_pool
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from src.utils import get_loops

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASET = 'fmnist'
EXP_CONFIG = 'dim_conf.yaml' 
ALGO = 'DiM' 
GAN = 'cgan'
IPC = 10 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__': 
    set_seed(42)
    
    config = get_config(EXP_CONFIG)[DATASET]
    trainset, testset = load_data(DATASET)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)  
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)  


    if ALGO == 'DiM':
        algo = DiM(
            dataset_name=DATASET, 
            opt=config, 
            trainloader=trainloader, 
            testloader=testloader, 
            gan_model_name=GAN,
            use_pretrained=True,
            pretrained_gan_path=f'pretrained/gan/{GAN}_{DATASET}.pth', 
            device=DEVICE)
    elif ALGO == 'D2M':
        algo = D2M(
            dataset_name=DATASET, 
            opt=config, 
            trainloader=trainloader, 
            testloader=testloader, 
            gan_model_name=GAN,
            use_pretrained=True,
            pretrained_gan_path=f'pretrained/gan/{GAN}_{DATASET}.pth', 
            device=DEVICE)
        
    elif ALGO == 'DC': 
        algo = GradientMatching(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config)
    elif ALGO == 'DSA':
        algo = DSA(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )
    elif ALGO == 'CAFE': 
        algo = CAFE(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )
    elif ALGO == 'DM': 
        algo = DistributionMatching(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )

    if GAN is not None:
        distill_time = algo.distillation() 
        algo.generate_sample(ipc=IPC)

        eval_model = get_random_model_from_model_pool(config)
        accuracy, elapsed_time, cpu_usage = algo.evaluate(eval_model, ipc=IPC)
    else: 
        outer_loop, inner_loop = get_loops(IPC)
        distill_time = algo.condensation(distillation_steps=config['distillation_steps'],
                                         outer_loop=outer_loop, 
                                         network_step=inner_loop)
        accuracy, elapsed_time, cpu_usage = algo.evaluate(config['eval_train_epochs'])

    print(f'{ALGO} - Image per class: {IPC}, Eval train time: {elapsed_time:.2f}s, CPU Usage: {cpu_usage:.2f}%, Final accuracy: {accuracy:.4f}%, Distillation time: {distill_time:.2f}s')
        
    