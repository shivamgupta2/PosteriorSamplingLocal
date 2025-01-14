from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

our_method = True
def load_yaml(file_path: str) -> dict:
    print(file_path)
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    denoising_fn = partial(sampler.denoise_sample_loop, model=model)
    annealed_langevin_fn = partial(sampler.annealed_langevin_loop, model=model)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'tilde_x', 'x_cond_tilde_x', 'denoising_progress', 'annealed_langevin_progress', 'annealed_langevin_res']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        elif not our_method: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            #sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
            
            sample = operator.transpose(y)

            noise_time = 300
            alphas_cumprod = sampler.get_alphas_cumprod()
            noise = torch.randn(ref_img.shape, device=device) * np.sqrt(1.0 - alphas_cumprod[noise_time])
            #tilde_x = ref_img * np.sqrt(alphas_cumprod[noise_time]) + noise
            tilde_x = sample * np.sqrt(alphas_cumprod[noise_time]) + noise
            tilde_x_scaled = tilde_x * np.sqrt(alphas_cumprod[noise_time])
            plt.imsave(os.path.join(out_path, 'tilde_x', fname), clear_color(tilde_x_scaled))
            x_cond_tilde_x = denoising_fn(tilde_x=tilde_x, noise_time=noise_time, record=True, save_root=out_path)
            plt.imsave(os.path.join(out_path, 'x_cond_tilde_x', fname), clear_color(x_cond_tilde_x))


            noise_var = noiser.sigma ** 2
            #annealed_vars = torch.zeros(1)
            #annealed_vars[0] = 1e4

            num_anneal_levels = 3000
            all_measurements = [y_n]
            annealed_vars = torch.zeros(num_anneal_levels)
            annealed_vars[0] = noise_var
            factor = 0.003
            for i in range(1, num_anneal_levels):
                annealed_vars[i] = annealed_vars[i-1] + factor * min(annealed_vars[i-1], 1)
                all_measurements.append(y_n + torch.randn_like(y_n) * torch.sqrt(annealed_vars[i] - annealed_vars[i-1]))

            annealed_vars = torch.flip(annealed_vars, [0])
            all_measurements.reverse()


            res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.000015, operator=operator, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path)
            #res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.00002, operator=operator, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path)
            #res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.00003, operator=operator, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path)
            plt.imsave(os.path.join(out_path, 'annealed_langevin_res', fname), clear_color(res))
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
            exit()
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

if __name__ == '__main__':
    main()
