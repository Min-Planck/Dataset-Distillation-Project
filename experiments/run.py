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
        print(f'{ALGO_NAME} - Image per class: {IPC}, Eval train time: {elapsed_time:.2f}s, CPU Usage: {cpu_usage:.2f}%, Final accuracy: {accuracy:.4f}%, Distillation time: {distill_time:.2f}s')

    else:
        train_generator_time = algo.train_generator()
        accuracy_1, distill_time_1, eval_time_1 = algo.evaluate(model='cnn', ipc=1)
        accuracy_10, distill_time_10, eval_time_10 = algo.evaluate(model='cnn', ipc=10)
        accuracy_50, distill_time_50, eval_time_50 = algo.evaluate(model='cnn', ipc=50)

        print(f'{ALGO_NAME} - Image per class: 1, Eval accuracy: {accuracy_1:.4f}%, Image per class: 10, Eval accuracy: {accuracy_10:.4f}%, Image per class: 50, Eval accuracy: {accuracy_50:.4f}%')       
        print(f'Distill time (1 ipc): {distill_time_1:.2f}s, Distill time (10 ipc): {distill_time_10:.2f}s, Distill time (50 ipc): {distill_time_50:.2f}s')
        print(f'Eval time (1 ipc): {eval_time_1:.2f}s, Eval time (10 ipc): {eval_time_10:.2f}s, Eval time (50 ipc): {eval_time_50:.2f}s')
        print(f'Train generator time: {train_generator_time:.2f}s')

