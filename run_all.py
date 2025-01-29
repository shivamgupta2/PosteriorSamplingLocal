import subprocess

# Configuration for the range and step size
start = 200
end = 1000
#step = 100
step = 100

# List to keep track of subprocesses
processes = []

# Loop through the range and launch each command as a subprocess
for i in range(start, start+step, step):
    i_begin = i
    i_end = i + step
    command = [
        "python3",
        "sample_condition.py",
        "--model_config=configs/model_config.yaml",
        "--diffusion_config=configs/diffusion_config.yaml",
        "--task_config=configs/inpainting_config.yaml",
        #"--task_config=configs/motion_deblur_config.yaml",
        #"--task_config=configs/gaussian_deblur_config.yaml",
        #"--task_config=configs/super_resolution_config.yaml",
        f"--i_begin={i_begin}",
        f"--i_end={i_end}",
    ]
    print(f"Starting process for i_begin={i_begin}, i_end={i_end}")
    # Start the subprocess
    processes.append(subprocess.Popen(command))

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All processes completed.")

