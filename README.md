# DDPM-for-meteo

This repository contains the source code of a denoising model based on probabilistic diffusion, implemented using Python. The model is designed for weather image denoising.

## Project structure

The project is structured as follows:

```bash
.
├── ddpm
│   ├── dataSet_Handler.py                  # Data manager
│   ├── ddpm_base.py                        # Basic model implementation/trainer/sampler
│   ├── conditionend_gaussian_diffusion.py  # Conditioned diffusion implementation
│   ├── sampler.py                          # Sampler implementation
│   └── trainer.py                          # Trainer implementation
└── utils
    ├── calculate_ensemble.py               # Utilities to calculate ensembles
    ├── config.py                           # Configuration manager
    ├── config_schema.json                  # Configuration schema, default values
    ├── distributed.py                      # Manager for the multi GPU distribution
    ├── guided_loss.py                      # Loss implementation for the conditioned diffusion
├── slurm/apptainer_stuff
│   ├── train_meluxina.sh                   # Launch a training job 
│   ├── sample_meluxina.sh                  # Launch a samplig job
├── main.py                                 # Code entry point
├── requirements.txt                        # Project dependencies
├── config_sample.yml                       # Exemple of sampling configuration
├── config_train.yml                        # Exemple of training configuration
└── README.md                               # This file
```

## Installation
You can install dependencies by running the following command:

```python
pip install -r requirements.txt
```

This code uses https://github.com/lucidrains/denoising-diffusion-pytorch.


## Directions for use

The main code is located in the `main.py` file and can be run in different modes:

- **Train** : Training mode.
- **Sample** : Sampling mode.

Execute the code with the following command:

```bash
python main.py --yaml_path [path_to_config_yaml] --debug [other_options]
```
Note that the path to the YAML configuration file is not mandatory, so the `--yaml_path` option is not mandatory. The default values are in `utils/config_schema.json`.
### Remark : 
You can also override the configuration options in the YAML file by specifying them directly on the command line. For example, to modify the batch size, you can add the option --batch_size [new_value].

## Available options

You can customize the behavior of this code by modifying/creating your own configuration. Here's a list of available options and their descriptions:

### General parameters :
- `mode` : The execution mode, you can choose between “Train” for training or “Sample” for sampling.
- `run_name` : The name of the training session or resuming directory.
- `batch_size` : The batch size (by default 32).
- `any_time` : The frequency of epochs at which the code saves the model and generates samples (by default 400).
- `model_path` : The path to the model to load and resume training if necessary (no path will start training from the beginning).
- `debug` : Active les journaux de débogage (par défaut : désactivé).
### Sampling parameters :
- `ddim_timesteps` : If not None, will sample from the ddim method with the specified number of time steps.
- `plot` : Activate to plot and save generated samples.
- `guided` : Path to conditioned data.
- `n_sample` : Number of samples to generate.
- `random_noise` : Use random noise for x_start in conditioned sampling.
### Data parameters :
- `data_dir` : Data directory.
- `v_i` : Variables number.
- `var_indexes` : Variable names (list).
- `crop` : Cropping parameters for the images.
- `auto_normalize` : Automatic normalization (disabled by default).
- `invert_norm` : Reverse normalization of image samples (disabled by default).
- `image_size` : Image size.
- `mean_file` : Path to the mean file.
- `max_file` : Path to the max file.
- `guiding_col` : Column to be used for conditioned sampling. Required when using conditioned mode.
- `csv_file` : Path to the labels csv file (Required when using conditioned mode).
- `dataset_config_file` : reprocessing configuration file path (only used if `rr` is one of the variables). 
### Model parameters :
- `scheduler` : Use a scheduler for the learning rate.
- `scheduler_epoch` : Number of epochs for the scheduler to ajust the learning rate (safe for resuming).
- `resume` : Resume from a checkpoint.
- `lr` : Learning rate.
- `adam_betas` : Bêtas for the Adam optimizer.
- `epochs` : Number of training steps.
- `beta_schedule` : Type of bêtas scheduler (cosine or linear).
### Monitoring parameters :
- `use_mlflow`: activate mlflow log
- `ml_tracking_uri`: path to log mlflow
- `ml_experiment_name`: mlflow experience name

- `wandbproject` : Name of the Wandb project.
- `use_wandb` : activate Wandb.
- `entityWDB` : Nom of the Wandb entity.

- `log_by_iteration`: `false` by default. If `true` : log the loss and the lr on every steps. 


### Pre-processing file parameters(exemple in config/rr_dataset_config.yml):
- `stat_folder` : Path to the directory that contains the normalization constants.
- `stat_version`: Gives an identifier for the normalization constants (by default : "rr")
- `rr_transform`: Parameters for the pe-processing of the precipitations
  - `log_transform_iteration` : How many times the log(1 + x) function is applied (0 to 2 typically)
  - `symetrization`: Random application (1/2) of a "-" sign before the samples of a strongly asymetric distribution in 0. False by default
  - `gaussian_std`: Threshold below which Gaussian noise is applied. Default 0 (no action).
- `normalization`: Normalization strategy
  - `type`: Choice of normalization format “mean” (data normalized mean 0, min/max between -0.95 and 0.95), “minmax”: data normalized min to -1 and max to 1, “quant”: data normalized by quantiles 1% and 99% (x <- -1 + 2(x-q01)/(q99-q01))
  - `per_pixel` : If normalization is performed on a per-pixel basis (requires spatialized constant files).default false
  - `for_rr`:
    - `blur_iteration`: Applies N successive Gaussian blurs if normalization is per pixel, on rr only. Default 1 

#### To resume a training, 2 possibilities :

- Start from a pre-trained model => pass it by `model_path`.
- Start from a pre-trained model AND continue in the same training folder => pass it by `model_path` AND use `resume`.

By default, the lr scheduler is `None` (constant learning rate). You must define `scheduler: OneCycleLR` and the `scheduler_epoch` (default: 150) to use PyTorch's `OneCycleLR`, and `scheduler: ReduceLROnPlateau` to use PyTorch's `ReduceLROnPlateau`. The scheduler is a PyTorch `OneCycleLR` scheduler. It is saved in the `.pt` file and is used to resume training, so it must be given the total number of training epochs.

## Exemples

1. Train the modem:

```python
python main.py --yaml_path config_train.yml --batch_size 64 --lr 0.0001
```

2. Test (Sample) the model

```python
python main.py --yaml_path config_sample.yml
```

3. Training with several GPUs 
```python
python torch.distributed.run --standalone --nproc_per_node gpu main.py --yaml_path config_sample.yml
```

4. Resume training from a checkpoint:

```python
python main.py --yaml_path config_train.yml --model_path checkpoints/checkpoint.pt --resume
```
warning, `--model_path` and `--resume` can be simply specified in the yaml file.

### Exemple of YAML config file:

```yaml
{
  # General parameters
  "mode": "Train",
  "run_name": "run_train",
  "batch_size": 4,
  "any_time": 25,

  # Sampling parameters
  "ddim_timesteps": 500,
  "plot": true,
  "sampling_mode": "simple",
  "n_sample": 4,
  "random_noise": true,

  # Data parameters
  "data_dir": "/path/to/your/data/",
  "csv_file": "your_data_labels.csv",
  "v_i": 3,
  "var_indexes": [ "u", "v", "t2m" ],
  "crop": [ 0,256,0,256 ],
  "invert_norm": false,
  "image_size": 256,
  "mean_file": "mean_data.npy",
  "max_file": "max_data.npy",
  "guiding_col": "your_guiding_column",

  # Model parameters
  "scheduler": true,
  "scheduler_epoch": 500,
  "resume": false,
  "epochs": 500,
  "beta_schedule": "linear",

  # Tracking parameters
  "use_mlflow": true, # activation mlflow log
  "ml_tracking_uri": "../mlruns", # path to log mlflow
  "ml_experiment_name": "ddpm", # experience name

  "wandbproject": "your_wandb_project",
  "use_wandb": true,
  "entityWDB": "your_entity"
}
```

If values are not specified in the configuration file, they will be replaced by default values or by overloading when the `main.py` file is called.

## Apptainer and Slurm

To run the previous commands on meluXina, it is necessary to be in the folder DE371_DiffusionModel. 
It is possible to run an interactive environment with the following commands : 
'''python
#Choose the duration of the job : 
#4 hours job : 
salloc -A p200177 -t 04:00:00 -p gpu -q short -N 1 -G 4 
#48 hours job :
salloc -A p200177 -t 48:00:00 -p gpu -q default -N 1 -G 4 

#Launch the container :
export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/" 
module load env/release/2023.1
module load env/staging/2023.1
module load Apptainer/1.2.4-GCCcore-12.3.0 
apptainer run --nv /project/home/p200177/DE_371/resources/apptainer_container/final_diffusion/container.sif
'''

Or it is possible to run a python file by using the files available in the slurm folder and filling the 'yaml_path' argument with the config you want to use. You can also choose the duration of the job. Then, juste use the following command :
'''python
sbatch file.sh
'''