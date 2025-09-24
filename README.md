# DDPM-for-meteo

This repository contains the source code of a denoising model based on probabilistic diffusion, implemented using Python. The model is designed for weather image denoising. The goal is to enrich the AROME-EPS dataset by generating members emulating the training data.
This code uses https://github.com/lucidrains/denoising-diffusion-pytorch as a basis.

The generation process consists in the progressive denoising of random gaussian noise images until they fit the targeted distribution of the training data.
This work allows two types of samples generation :
  - unconditional generation : The model generates random samples relevant to the training set (the climatology)
  - conditional generation : The model is constrained with parameters relevant from an ensemble, in order to generate members from this specific ensemble.

## Project structure

The project is structured as follows:

```bash
.
├── ddpm
│   ├── dataSet_Handler.py                  # Data manager
│   ├── ddpm_base.py                        # Basic model implementation/trainer/sampler
│   ├── conditionend_gaussian_diffusion.py  # Conditioned diffusion implementation
│   ├── denoising_diffusion_pytorch.py      # Core model script
│   ├── elucidated_diffusion.py             # Fast sampler implementation
│   ├── normalize.py                        # Utilitary file for normalization
│   ├── special_transforms.py               # Specific normalization for certain variables
│   ├── sampler.py                          # Sampler implementation
│   └── trainer.py                          # Trainer implementation
└── utils
    ├── calculate_ensemble.py               # Utilities to calculate ensembles
    ├── config.py                           # Configuration manager
    ├── config_schema.json                  # Configuration schema, default values
    ├── distributed.py                      # Manager for the multi GPU distribution
    └── guided_loss.py                      # Loss implementation for the conditioned diffusion
├── slurm/apptainer_stuff
│   ├── train_meluxina.sh                   # Launch a training job 
│   ├── sample_meluxina.sh                  # Launch a samplig job
│   └── unbias_sample_meluxina.sh           # Launch a job to unbias the generated samples
├── main.py                                 # Code entry point
├── requirements.txt                        # Project dependencies
├── config                                  # config exemples
│   ├── ed                                    # 3var u, v, t2m
│   └── ed_4var                               # 4var rr, u, v, t2m
└── README.md                               # This file
```

## Installation
You can install dependencies by running the following command:

```python
pip install -r requirements.txt
```

## The AROME-EPS Dataset

The dataset comprises 516 AROME ensemble forecasts covering the period from June 15th, 2020, to November 12th, 2021. Each ensemble forecast is composed of 16 members and includes lead times at 1-hour intervals, ranging up to 45 hours. It follows that [516x45x16=371520]() individual samples are available for training if each members of the enseble at a given lead time is considered individually.

The data is restricted to a region encompassing the south and center of France with a resolution of [256x256]. Four variables are here considered: the precipitation (rr in mm/h) the horizontal (u) and vertical (v) in m/s components of the wind speed vector at 10 meters and the temperature at 2 meters (t2m)in K. Each individual sample can be conceptualized as a tensor with 4 channels, a width of 256 and a height of 256 [4, 256, 256].

To efficiently load and organize the dataset, a metadata CSV file is utilized. The file structure is illustrated below:

| Name          | Importance | PosX | PosY | Date       | LeadTime | Member |
|---------------|------------|------|------|------------|----------|--------|
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |
| _sample1440   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 0      |
| _sample1441   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 1      |
| _sample1442   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 2      |
| _sample1443   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 3      |
| _sample1444   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 4      |

- **`Name`**: A unique identifier for each sample.
- **`Importance`**: Importance level.
- **`PosX` and `PosY`**: Size of the image [**TO BE CONFIRMED**]
- **`Date`**: Date of the ensemble forecast.
- **`LeadTime`**: Lead time in hours.
- **`Member`**: Member index within the ensemble.

This metadata file plays a crucial role in loading the dataset efficiently and ensuring the proper association of each sample with its corresponding attributes. Please update the file path in your code to reflect the location of your metadata CSV file.

The files containing the values for normalization can be generated in the subfolder **preprocessing/Preprocess_datas_IS_split**.  

Each folder presented here includes its own description file.

## Directions for use

The main code is located in the `main.py` file and can be run in different modes:

- **Train** : Training mode.
- **Sample** : Sampling mode.

Execute the code with the following command:

```bash
python main.py --yaml_path [path_to_config_yaml]
```
Note that the path to the YAML configuration file is not mandatory, so the `--yaml_path` option is not mandatory. The default values are in `utils/config_schema.json`. More generally this file contains the exhaustive list and the description of all the parameters.

### Remark : 
You can also override the configuration options in the YAML file by specifying them directly on the command line. For example, to modify the batch size, you can add the option --batch_size [new_value].

## Available options

You can customize the behavior of this code by modifying/creating your own configuration. Config exemples are available in 

#### To resume a training, 2 possibilities :

- Start from a pre-trained model => pass it by `model_path`.
- Start from a pre-trained model AND continue in the same training folder => pass it by `model_path` AND use `resume`.

By default, the lr scheduler is `None` (constant learning rate). You must define `scheduler: OneCycleLR` and the `scheduler_epoch` (default: 150) to use PyTorch's `OneCycleLR`, and `scheduler: ReduceLROnPlateau` to use PyTorch's `ReduceLROnPlateau`. The scheduler is a PyTorch `OneCycleLR` scheduler. It is saved in the `.pt` file and is used to resume training, so it must be given the total number of training epochs.

## Exemples

1. Train the model:

```python
python main.py --yaml_path your_training_config.yml --batch_size 64 --lr 0.0001
```

2. Test (Sample) the model:

```python
python main.py --yaml_path your_sampling_config.yml
```

3. Resume training from a checkpoint:

```python
python main.py --yaml_path config_train.yml --model_path checkpoints/checkpoint.pt --resume
```
Functional exemples of yaml files are available in the config folder.

!WARNING! Don't forget to change the paths in config file.
