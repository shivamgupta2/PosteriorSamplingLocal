import numpy as np
import torch
from guided_diffusion.gaussian_diffusion import create_sampler

from functools import partial
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import os
import argparse
import yaml

annealed = False
def load_yaml(file_path: str) -> dict:
    print(file_path)
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def gaussian_unconditional_eps(x, t):
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    print('here')

    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    diffusion_config = load_yaml(args.diffusion_config)
   

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    annealed_langevin_fn = partial(sampler.annealed_langevin_loop, model=gaussian_unconditional_eps)

    num_samples = 1000
    sigma = 0.3
    x = torch.randn(num_samples)

    y = x + torch.randn(num_samples) * sigma
    x = x[:, None].expand(-1, num_samples)
    y = y[:, None].expand(-1, num_samples)


    if annealed:
        num_anneal_levels = 1000
        all_measurements = [y]
        annealed_vars = torch.zeros(num_anneal_levels)
        annealed_vars[0] = sigma ** 2
        factor = 0.1
        for i in range(1, num_anneal_levels):
            annealed_vars[i] = annealed_vars[i-1] + factor * min(annealed_vars[i-1], 1)
            all_measurements.append(y)
            #all_measurements.append(y + torch.randn_like(y) * torch.sqrt(annealed_vars[i] - annealed_vars[i-1]))

        annealed_vars = torch.flip(annealed_vars, [0])
        all_measurements.reverse()
        num_steps = 100
    else:
        all_measurements = [y]
        annealed_vars = torch.zeros(1)
        annealed_vars[0] = sigma ** 2
        num_steps=100000

    print(annealed_vars)
    res = annealed_langevin_fn(x_cond_tilde_x = torch.randn_like(x), all_measurements=all_measurements, anneal_vars=annealed_vars, num_anneal_steps=num_steps, step_size=0.0001, operator=lambda x: x, transpose=lambda x:x, alpha_cumprod=0, record=False, save_root='no_annealing')
      
if __name__ == '__main__':
    main()
