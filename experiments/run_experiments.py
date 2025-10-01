from config import get_config
from src import DiM, D2M, load_data
from models.model_pooL import get_random_model_from_model_pool
from src.algo.dc import GradientMatching
import torch
from torch.utils.data import DataLoader

DATASET = 'fmnist'
EXP_CONFIG = 'dim_conf.yaml' 
ALGO = 'DiM' 
GAN = 'acgan'
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
            pretrained_gan_path=f'pretrained_models/gan/{GAN}_{DATASET}_generator.pth', 
            device=DEVICE)
    elif ALGO == 'D2M':
        algo = D2M(
            dataset_name=DATASET, 
            opt=config, 
            trainloader=trainloader, 
            testloader=testloader, 
            gan_model_name=GAN,
            use_pretrained=True,
            pretrained_gan_path=f'pretrained_models/gan/{GAN}_{DATASET}_generator.pth', 
            device=DEVICE)
        
    elif ALGO == 'DC': 
        algo = GradientMatching(
            model_name='ConvNet',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config)
        
    if GAN is not None:
        algo.distillation() 
        algo.generate_sample(ipc=IPC)

        eval_model = get_random_model_from_model_pool(config)
        accuracy = algo.evaluate(eval_model, ipc=IPC)
    else: 
        algo.distillation(100, 20)
        accuracy = algo.evaluate(50)


    print(f'{ALGO} - Image per class: {IPC} - Final accuracy: {accuracy:.2f}%')
        
    