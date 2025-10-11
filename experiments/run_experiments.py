import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src import DiM, D2M, GradientMatching, DistributionMatching, DSA, CAFE, load_data
from src.models import get_random_model_from_model_pool
import torch
from torch.utils.data import DataLoader

DATASET = 'fmnist'
EXP_CONFIG = 'dim_conf.yaml' 
ALGO = 'DiM' 
GAN = 'cgan'
IPC = 10 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__': 
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
        algo.distillation() 
        algo.generate_sample(ipc=IPC)

        eval_model = get_random_model_from_model_pool(config)
        accuracy = algo.evaluate(eval_model, ipc=IPC)
    else: 
        if ALGO != 'CAFE': 
            algo.condensation(config['distillation_steps'], config['network_step'])
        else: 
            algo.condensation(config['network_step'])
        accuracy = algo.evaluate(config['eval_train_epochs'])

    print(f'{ALGO} - Image per class: {IPC} - Final accuracy: {accuracy:.4f}%')
        
    