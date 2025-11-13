import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.utils import get_configs, set_seed, load_data, get_loops
from src.algo import * 

ALGO_NAME = 'dc'
DATASET = 'fmnist'
GAN = True if ALGO_NAME in ['dim', 'dcm'] else None
IPC = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seed(42)
    config = get_configs(f'{ALGO_NAME}_{DATASET}')
    print(config)
    trainset, testset = load_data(DATASET)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)  
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)  

    if ALGO_NAME == 'dc':
        algo = GradientMatching(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config)
    elif ALGO_NAME == 'dsa':
        algo = DSA(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )
    elif ALGO_NAME == 'cafe': 
        algo = CAFE(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )
    elif ALGO_NAME == 'dm': 
        algo = DistributionMatching(
            model_name='cnn',
            trainset=trainset, 
            testloader=testloader, 
            device=DEVICE,
            ipc=IPC,
            opt=config
        )
    elif ALGO_NAME == 'dim':
        algo = DiM(
            trainloader=trainloader, 
            testloader=testloader,
            device=DEVICE,
            opt=config
        )

    
    if not GAN: 
        outer_loop, inner_loop = get_loops(IPC)
        distill_time = algo.condensation(distillation_steps=config['distillation_steps'],
                                         outer_loop=outer_loop, 
                                         network_step=inner_loop)
        accuracy, elapsed_time, cpu_usage = algo.evaluate(config['eval_train_epochs'])
    else:
        distill_time = algo.train_generator()
        accuracy = algo.evaluate(model='cnn', ipc=IPC)
        elapsed_time, cpu_usage = 0, 0
    print(f'{ALGO_NAME} - Image per class: {IPC}, Eval train time: {elapsed_time:.2f}s, CPU Usage: {cpu_usage:.2f}%, Final accuracy: {accuracy:.4f}%, Distillation time: {distill_time:.2f}s')
