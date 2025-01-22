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
use_existing_dps_recon = True
def load_yaml(file_path: str) -> dict:
    print(file_path)
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_annealed_measurements_and_vars(num_anneal_levels, factor, y_n, noise_var):
    all_measurements = [y_n]
    annealed_vars = torch.zeros(num_anneal_levels)
    annealed_vars[0] = noise_var
    for anneal_ind in range(1, num_anneal_levels):
        if annealed_vars[anneal_ind] > 0.0030:
            annealed_vars[anneal_ind] = annealed_vars[anneal_ind-1] + factor * min(annealed_vars[anneal_ind-1], 1)
        else:
            annealed_vars[anneal_ind] = annealed_vars[anneal_ind-1] + (1.1*factor) * min(annealed_vars[anneal_ind-1], 1)
        #annealed_vars[i] = annealed_vars[i-1] * (1 + factor * min(annealed_vars[i-1] ** 2, 1.0/(annealed_vars[i-1] ** 2)))
        all_measurements.append(y_n + torch.randn_like(y_n) * torch.sqrt(annealed_vars[anneal_ind] - annealed_vars[anneal_ind-1]))

    annealed_vars = torch.flip(annealed_vars, [0])
    all_measurements.reverse()
    return all_measurements, annealed_vars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--i_begin', type=int, default=0)
    parser.add_argument('--i_end', type=int, default =-1)
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
    for img_dir in ['input', 'recon', 'progress', 'label', 'tilde_x', 'x_cond_tilde_x', 'denoising_progress', 'annealed_langevin_progress', 'annealed_langevin_res', 'annealed_langevin_res_unclamped', 'dps_unclamped']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    print(data_config)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    batch_size = 100
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0, train=False)
    if use_existing_dps_recon:
        input_dataset = get_dataset(name='ffhq_input', root='results/super_resolution/input', transforms=transform)
        dps_dataset = get_dataset(name='ffhq_dps', root='results/super_resolution/recon', transforms=transform)
        input_loader = get_dataloader(input_dataset, batch_size=batch_size, num_workers=0, train=False)
        dps_loader = get_dataloader(dps_dataset, batch_size=batch_size, num_workers=0, train=False)
        loader = zip(loader, input_loader, dps_loader)



    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    params_list = [(1000, 0.009, 1, 5 * 1e-7), (1000, 0.009, 1, 2.5 * 1e-6), (1000, 0.009, 1, 5 * 1e-6), (1000, 0.009, 1, 7.5 * 1e-6)]
    for params in params_list:
        img_dir = f'annealed_langevin_res_{params}'
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        img_dir = f'annealed_langevin_res_unclamped_{params}'
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Do Inference
    for i, ref_img in enumerate(loader):
        if args.i_end != -1:
            if i < args.i_begin or i >= args.i_end:
                continue
        logger.info(f"Inference for image {i}")
        #fname = str(i).zfill(5) + '.png'
        if use_existing_dps_recon:
            ref_img = (ref_img[0].to(device), ref_img[1].to(device), ref_img[2].to(device))
        else:
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
            # Sampling
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        elif not our_method: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
            # Sampling
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        elif use_existing_dps_recon:
            print('using existing DPS recons')
            y_n = ref_img[1]
            noise_var = noiser.sigma ** 2
            #params = (1000, 0.009, 1, 0.00002)
            #params = (1000, 0.009, 1, 0.000001)
            for params in params_list:
                num_anneal_levels, factor, num_anneal_steps, step_size = params
                all_measurements, annealed_vars = get_annealed_measurements_and_vars(num_anneal_levels, factor, y_n, noise_var)
                print(annealed_vars)
                sample = ref_img[2]

                
                noise_time = 100
                alphas_cumprod = sampler.get_alphas_cumprod()
                noise = torch.randn(sample.shape, device=device) * np.sqrt(1.0 - alphas_cumprod[noise_time])
                tilde_x = sample * np.sqrt(alphas_cumprod[noise_time]) + noise
                tilde_x_scaled = tilde_x * np.sqrt(alphas_cumprod[noise_time])

                x_cond_tilde_x = denoising_fn(tilde_x=tilde_x, noise_time=noise_time, record=False, save_root=out_path)

                res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=num_anneal_steps, step_size=step_size, operator=operator.forward, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=False, save_root=out_path)
                for j in range(len(res)):
                    cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                    plt.imsave(os.path.join(out_path, f'annealed_langevin_res_unclamped_{params}', cur_fname), clear_color(res[j].unsqueeze(0)))

                res = torch.clamp(res, -1, 1)
                for j in range(len(res)):
                    cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                    print('writing annealed langevin result, file name:', cur_fname)
                    plt.imsave(os.path.join(out_path, f'annealed_langevin_res_{params}', cur_fname), clear_color(res[j].unsqueeze(0)))

        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)


            noise_var = noiser.sigma ** 2

            #params = (500, 0.015, 0.00003)
            #params = (500, 0.014, 1, 0.00003)
            #params = (500, 0.014, 2, 0.000015)
            #params = (700, 0.011, 1, 0.000015)
            #params = (1000, 0.009, 1, 0.000015)


            params = (1000, 0.009, 1, 0.00002)

            #params = (2000, 0.0045, 1, 0.00001)
            #params = (4000, 0.0022, 1, 0.000005)
            #annealed_vars = torch.zeros(1)
            #annealed_vars[0] = 1e4

            num_anneal_levels, factor, num_anneal_steps, step_size = params
            #num_anneal_levels = 600
            #num_anneal_levels = 500
            #num_anneal_levels = 1000
            #num_anneal_levels = 1500
            #num_anneal_levels = 2000
            #num_anneal_levels = 3000
            #num_anneal_levels = 5000
            #factor = 0.01
            #factor = 0.007
            #factor = 0.0065
            #factor = 0.0045
            #factor = 0.0065
            #factor = 0.007
            #factor = 0.0075
            #factor = 0.015
            #factor = 0.014
            #factor = 16.5
            #factor = 81

            all_measurements, annealed_vars = get_annealed_measurements_and_vars(num_anneal_levels, factor, y_n, noise_var)
            print(annealed_vars)

            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)
            
            for j in range(len(sample)):
                cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                plt.imsave(os.path.join(out_path, 'recon', cur_fname), clear_color(sample[j].unsqueeze(0)))
            #sample = operator.transpose(y)

            #noise_time = 300
            noise_time = 100
            alphas_cumprod = sampler.get_alphas_cumprod()
            noise = torch.randn(ref_img.shape, device=device) * np.sqrt(1.0 - alphas_cumprod[noise_time])
            #tilde_x = ref_img * np.sqrt(alphas_cumprod[noise_time]) + noise
            tilde_x = sample * np.sqrt(alphas_cumprod[noise_time]) + noise
            del sample
            del x_start

            tilde_x_scaled = tilde_x * np.sqrt(alphas_cumprod[noise_time])
            for j in range(len(tilde_x_scaled)):
                cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                print('here cur fname:', cur_fname)
                plt.imsave(os.path.join(out_path, 'tilde_x', cur_fname), clear_color(tilde_x_scaled[j].unsqueeze(0)))
            del tilde_x_scaled
            x_cond_tilde_x = denoising_fn(tilde_x=tilde_x, noise_time=noise_time, record=False, save_root=out_path)
            for j in range(len(x_cond_tilde_x)):
                cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                print('here cur fname:', cur_fname)
                plt.imsave(os.path.join(out_path, 'x_cond_tilde_x', cur_fname), clear_color(x_cond_tilde_x[j].unsqueeze(0)))

            #res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.00001, operator=operator.forward, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path, normalize_at_end_of_step=False)
            #res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.00002, operator=operator.forward, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path, normalize_at_end_of_step=False)
            #res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=1, step_size=0.00003, operator=operator.forward, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=True, save_root=out_path)
            res = annealed_langevin_fn(x_cond_tilde_x=x_cond_tilde_x, all_measurements=all_measurements, anneal_vars = annealed_vars, num_anneal_steps=num_anneal_steps, step_size=step_size, operator=operator.forward, alpha_cumprod=alphas_cumprod[0], transpose=operator.transpose, record=False, save_root=out_path)

            for j in range(len(res)):
                cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                plt.imsave(os.path.join(out_path, 'annealed_langevin_res_unclamped', cur_fname), clear_color(res[j].unsqueeze(0)))

            res = torch.clamp(res, -1, 1)
            for j in range(len(res)):
                cur_fname = str(i * batch_size + j).zfill(5) + '.png'
                plt.imsave(os.path.join(out_path, 'annealed_langevin_res', cur_fname), clear_color(res[j].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'input', cur_fname), clear_color(y_n[j].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'label', cur_fname), clear_color(ref_img[j].unsqueeze(0)))
            #exit()
         

if __name__ == '__main__':
    main()
