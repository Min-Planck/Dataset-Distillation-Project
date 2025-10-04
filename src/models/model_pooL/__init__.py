from .convnet import CNN
from .resnet import ResNet18, ResNet34
from .lenet import LeNet
from .alexnet import AlexNet
from .mlp import MLP

import random

def get_random_model_from_model_pool(opt):
    idx = random.randint(1, 6)
    if idx == 1: 
        model = CNN(opt)
    elif idx == 2:
        model = ResNet18(opt)
    elif idx == 3:
        model = ResNet34(opt)
    elif idx == 4:
        model = AlexNet(opt)
    elif idx == 5:
        model = LeNet(channel=opt['channel'], num_classes=opt['n_classes'])
    elif idx == 6: 
        model = MLP(channel=opt['channel'], im_size=opt['im_size'], num_classes=opt['n_classes'])

    return model

    
def get_model_by_name(model_name, opt): 

    if model_name.lower() == 'cnn':
        model = CNN(opt)
    elif model_name.lower() == 'mlp': 
        model = MLP(channel=opt['channel'], im_size=opt['im_size'], num_classes=opt['n_classes'])
    elif model_name.lower() == 'lenet':
        model = LeNet(channel=opt['channel'], num_classes=opt['n_classes'])
    elif model_name.lower() == 'alexnet':
        model = AlexNet(channel=opt['channel'], num_classes=opt['n_classes'])
    elif model_name.lower() == 'resnet18':
        model = ResNet18(opt)
    elif model_name.lower() == 'resnet34':
        model = ResNet34(opt)
    else: 
        raise ValueError(f"Model {model_name} not recognized. Available models: cnn, mlp, lenet, alexnet, resnet18, resnet34.")

    return model
    
    