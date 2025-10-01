import yaml

def get_config(conf): 
    with open(f'config/{conf}', 'r') as f:
        config = yaml.safe_load(f)
    return config