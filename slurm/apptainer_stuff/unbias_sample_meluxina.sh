#!/bin/bash -l
#SBATCH -J DE371_diffusion_sampling
#SBATCH -A p200177
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
#SBATCH --qos=default

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
    python3 main_unbias.py \
    --real_data_dir='path/to/original/data' \
    --gen_data_dir='path/to/generated/data/' \
    --output_dir='path/to/output/dir/' \
    --dates_file='Large_lt_val_labels_ens.csv' \
    --date_start='2020-07-01' \
    --date_stop='2021-07-01' \