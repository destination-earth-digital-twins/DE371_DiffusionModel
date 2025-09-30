# Some specific configurations

Here we detail some parameters to set to allow specific functionalities of the code

## Train an unconditional model on full AROME domain
In the yaml file, set:
```
"n_conditions": 0
```
Also, to achieve a proper training, it is important to condition the model by the orography. To do so, set :

```
"orography_conditioning": true
```
and :
```
"sampling_mode": "conditioned_input"
```
and :
```
"path_to_orography" : "/project/home/p200177/DE_371/avritj/orography_mirror_filled.npy" #TODO : put this file into the project resources
```

Then, to train on big domain : 

```
"domain_type" : "big" #probably going to be modified because redundant with crop
```

To replace invalid values : 
```
"training_configuration": "mirror" #gonna get removed as this is the best filling method so far
```


```
"model_used": "unet" to train with the unet (other choices avalaible, see the config_schema.json file)
```

if training with the U-Net, we advise setting "use_AMP" and "compile_model" to True, as it fastens greatly the training. 

The "crop" values must follow this rule :


```
crop[1] - crop[0] #divisible by 8
crop[3] - crop[2] #divisible by 8
```



Then launch yout training as usual:
```python
python main.py --yaml_path your_training_config.yml
```

## Sample using SDEdit 
From [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) by Meng et al.
This sampling mode allows to generate conditional samples from an unconditional model.

In the yaml file, set:
```
"sampling_mode": "conditioned_sdedit"
"model_path": "path/to/your/UNCONDITIONAL/model/best.pt
```
Then launch yout sampling as usual:
```python
python main.py --yaml_path your_training_config.yml
```
## Train and Sample using SEEDS
From [SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models](https://arxiv.org/abs/2306.14066) by Li et al.
This  mode allows to generate conditional samples from a conditional model. The model needs to be trained beforehand.

Different conditions can be specified in the yaml file. The model accepts the following conditions : 
- The number of conditioning members, from 0 to 15 :
```
"n_conditions": 1
```
- The ensemble spatial mean :
```
"mean_conditioning": true
```
- The normalized ensemble spatial variance :
```
"var_conditioning": true
```
!WARNING! the same conditions must be same for both the training AND the sampling