# Editing yaml for FMFastSim

A yaml file for model training contains the following information

```
train_model.yaml
├── experiment          : name of the experiments and define check point and validation directories 
├── gpu_info            : use gpu or cpu, and, if using gpu, use distributed data parallel or not. 
├── model_info          : contains information about the model 
│   ├── gen_param       : define the generative frameworks, e.g., AE,VAE,GAN, and Diffusion Model, and its parameters
│   └── network_param   : define the model architecture
├── train_info          : contains training parameters, such as learning rate, number of epochs, optimizer, and so on 
├── data_info           : which data loader to use 
└── scale_info          : define a transformation of the raw data
```

## experiment

This section contains the name and other information of the training run. Here is an example
```
experiment:
    study_name: Diffusion
    run_name:   MixerTF_Log
    checkpoint_dir: /dccstor/kyeo_data/chk_point
    validation_dir: /dccstor/kyeo_data/validation
``` 

`checkpoint_dir` is the directory to save the intermeidate models during the training and `validation_dir` is the location of the validation plots. 
The following directories will be automatically created (if not already exists).
```
{checkpoint_dir}/{study_name}/{run_name}
{validation_dir}/{study_name}/{run_name}
```

## gpu_info

This section is to use GPU and/or Distributed Data Parallel (DDP) Training.
```
gpu_info:
    use_gpu: True
    use_ddp: False
```
Set `use_ddp: True` to use DDP for multiple GPUs. To use DDP, execute the script using `torchrun`.
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node={number of gpu} --rdzv_endpoint="localhost:8800" SRC_DIR/main/train.py train_model.yaml
```

## model_info

`model_info` contains information about the neural network architecture as well as a generative model. Here is an example

```
model_info:
    model_type: VAE             #generative frameworks: AE,MLE,VAE,GAN,Diffusion
    network_type: MixerTF       #types of the neural network: Mixer, MixerTF, PatchTSMixer
    save_model: True            #save the itermediate models
    load_file: null             #path and file name of a trained network, if loading a pre-trained model.
    save_file: VAE_MixTF.pt     #path and file name to save the trained model.
                                #If null, the model is saved in chkeckpoint_dir using the default name
    gen_param:                  #paramters for generative framework
        kl_coef: 1.0
    network_param:
        dim_r: 18
        dim_a: 50
        dim_v: 45
        dim_c: 4
        dim_r_enc: [18,18, 9]
        dim_a_enc: [50,25,25]
        dim_v_enc: [45,45,30]
        dim_r_dec: [ 9,18,18,18]
        dim_a_dec: [25,25,50,50]
        dim_v_dec: [30,45,45,45]
        num_enc_heads:  [5,5]
        num_dec_heads:  [5,5,5]
        mixer_dim:  3
        mlp_ratio:  3
        mlp_layers: 3
        res_conn: True
        gated_attn: False
        activation: SiLU
        final_layer: True
        variational: True  #Use Gaussian latent state or not
```
`gen_param` depends on the type of the generative framework defined in `model_type`. For the details, see [here](../core/generative_model/README.md)

`network_param` is the parameters to build a neural network chosen in `network_type`. Here, `dim_r, dim_a, dim_v` are the dimensions in the radial ( $r$ ), azimuthal ( $\phi$ ), and vertical ( $z$ ) directions. `dim_c` is the dimension of the conditioning variables. 

All the `network_param` depends on the selected model, except `variational`. All the models are expected to have `variational` as an input argument. See [here](../core/layers/README.md).

## train_info

This section contains the training related parameters. For example,

```
train_info:
    optimizer: Adam
    epochs: 2000
    learning_rate: 1.e-4
    batch_size: 128
    lr_scheduler:
        scheduler: scheduled  #either 'scheduled' or 'on_plateau'
        milestones: [10,20,40,80,160,320,640]
        gamma: 0.5
    save_freq: 20
    plot_freq: 20
    plot_config:
        - [70,  64, 'SiW']
        - [70, 128, 'SiW']
        - [70, 256, 'SiW']
```

The training code has two types of learning rate schedulers, `MultiStepLR` and `ReduceLROnPlateau`. If `schedule: scheduled`, `MultiStepLR` is used. And, set it to `on_plateau` to use `ReduceLROnPlateau`. For `schedule: scheduled`, we need to set
```
    lr_scheduler:
        scheduler: scheduled  
        milestones: [10,20,40,80,160,320,640] #epoch number to reduce LR
        gamma: 0.5                            #LR_new = LR_old*gamma
```
For `schedule: on_plateau`
```
    lr_scheduler:
        scheduler: on_plateau  
        factor: 0.5   #LR_new = LR_old*factor
        patience: 10  #how many epochs to wait before reducing LR

```

`save_freq` defines when to save the intermediate models. The intermediate models are saved in `checkpoint_dir` in every `save_freq` epochs.

`plot_freq` is the interval between generating validation plots, which are saved in `validation_dir`. And `plot_config` is a list of conditioning variables to make the validation plots

## data_info

This section contains data specific parameters, e.g.,

```
data_info:
    dataloader: 'shower_dataset_small'
    root_path: '/datasets/CERN_Fast_Shower'
    geo: ["SiW"]
    angles: [70]
    energies: [64,128,256]
    data_split: [0.95,0.03,0.02]   #fractions of train, validation, test sets
    max_num_events: 100000
```

`dataloader` is the name of the dataloader. `dataloader` should be under `{MAIN}/dataloader`. For example, in this case, `dataloader` looks for a file `{MAIN}/dataloader/shower_dataset_small.py`

`root_path` is the location of the actual data file. See an example dataloader [here](../dataloader/shower_dataset_small.py)

`geo, angles, energies` are conditioning variables to train the model.

`max_num_events` is the maximum number of events to load. It is used only when the memory is limited.

## scale_info

Here, we define what kind of data transformtion to use. For example,

```
scale_info:
    scale_by_energy: True
    scale_method: 'lin_trans'
    scale_param:
        bias:  0.0
        scale: 0.2
```

`scale_by_energy` indicates to scale the raw data by an incident energy. If `scale_by_energy: True`, the data is first transformed by dividing corresponding incident energies.

`scale_methd` has four options, `lin_trans, log_trans, logit_trans,` and `identity`. `identity` means the data is not transformed.

`lin_trans` is a linear transformation $\hat{x} = (x+bias)/scale$

`log_trans` is a log transformation. It has three parameters
```
scale_info:
    scale_method: 'log_trans'  
    scale_param:
        mag:   1.0
        bias:  1.e-5
        scale: 0.05
```
The data is transformed as $\hat{x} = mag * ( \log(x+bias) -\log(scale) )$.

`logit_trans` has the same three parameters as `log_trans`: `mag, bias, scale`. It first makes a tanh transformation to make sure the variable is less than 1, i.e., $\tilde{x} = \tanh((x+bias)/scale)$. Then, a logit transformatin is made, $\hat{x} = \log(\tilde{x}) - \log(1-\tilde{x})$.

Note that all these transformations are made on the data, after `scale_by_energy` transformation is applied. For the details, see the source code [here](../utils/data_scaling.py).
