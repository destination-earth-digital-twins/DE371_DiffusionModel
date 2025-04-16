#!/bin/bash -l
#SBATCH -J DE371_diffusion
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --time=6:00:00
#SBATCH --qos=short

export TORCH_DISTRIBUTED_DEBUG=INFO 
export OMP_NUM_THREADS=4
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export NCCL_ASYNC_ERROR_HANDLING=1
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0
module load NVHPC
module load GCC

export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"

source .env
# Echo des commandes lancees
set -x

# Ensure the logs directory exists or create it if not
# if [ ! -d "$LOG_DIR" ]; then
#   mkdir "$LOG_DIR"
# fi

apptainer run --nv /project/home/p200177/DE_371/resources/apptainer_container/final_diffusion/container.sif \
    python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 ./main.py \
<<<<<<< HEAD
    --yaml_path="/home/users/u101957/DE371_DiffusionModel/config/ed/config_sample_conditioned_ED_val.yml"
=======
    --yaml_path="path/to/sample/config.yml"
>>>>>>> main
