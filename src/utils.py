import numpy as np
import torch
#import random

def set_seed(seed):
    '''
    Set random seed for reproducibility.
    
    Args:
        seed: int
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    #random.seed(seed)
