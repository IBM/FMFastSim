# Foundation Model for Fast Shower Simulation (FMFastSim)

FMFastSim aims to learn and simulate a shower event for a wide range of experimental conditions, using generative AI models.

## Directory structure

```bash
main
├── core                  : contains main model handler 
│   ├── generative_model  : contains codes for generative frameworks, e.g., AE,VAE,GAN, and Diffusion Model
│   └── layers            : contains codes for base neural networks, e.g., Transformers, PatchTSMixer, ...
├── dataloader            : contains data handler and data loader
├── utils                 : contains data scaling, and postprocessing functions, such as validation and plotting functions
└── yaml                  : example yaml files for model training, validation, and shower simulation
```

## Getting Started
FMFastSim requires the following libraries:
```
h5py==3.9.0
matplotlib==3.5.1
numpy==1.26.4
plotly==5.17.0
scipy==1.12.0
torch==2.0.1
transformers==4.37.2
```

FMFastSim doesn't need to be installed. Simply, after cloning the repo on SRC_DIR, move to the main directory, and install the required libraries.

```
cd SRC_DIR/main
pip install -r requirements.txt
```

## Sample Full simulation dataset

A Sample full simulation dataset can be downloaded from/linked to [Zenodo](https://zenodo.org/record/6082201#.Ypo5UeDRaL4).

## Training

First, edit a yaml file using a text editor. Yaml file contains the following information

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
Each section of the yaml file is explained in [here](yaml/README.md).

Make a soft link of the training script `main/train.py`
```
ln -s SRC_DIR/main/train.py train.py
```

Then, execute the traininig script
```
python train.py train_model.yaml
``` 
Or, simply execute the training script in the main directoy
```
python SRC_DIR/main/train.py train_model.yaml
```

If there are more than one GPU, and if `use_ddp: True` under `gpu_info`, the code will automatically use all the GPUs visible for a distributed data parallel training. To use the distributed data parallel training, execute
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node={number of gpu} --rdzv_endpoint="localhost:8800" SRC_DIR/main/train.py train_model.yaml
```


## Validation

In order to validate the trained MLFastSim against the full simulation, use `validate.py` code.
First, edit a yaml file. A validation yaml file has the following structure

```
valid_model.yaml
├── gpu_info     : use a gpu or compute on cpu 
├── model_info   : define the location of the trained model 
├── valid_info   : conditioning variables to do validations and location of the validation plots  
└── data_info    : which data loader to use 
```

All the information to define the generative framework and the neural network architecture, such as `gen_param` and `network_param`, are saved with the trained model. In the validation and simulation time, we can simply direct the location of the trained model, then the code will automatically use the saved parameters to load the trained model.

Similar to Training, the validation can be executed either by using a soft link or directly executing the validation code in the main directory
```
python SRC_DIR/main/validate.py valid_model.yaml
```

## ML shower generation

Generating Shower events are done similar to the validation. First, edit a yaml file.

```
generate_model.yaml
├── gpu_info     : use a gpu or compute on cpu 
├── model_info   : define the location of the trained model 
├── gen_info     : conditioning variables for the event and location to save the generated events.  
└── data_info    : which data loader to use 
```

While, in the generation, no data is loaded, still we need to define the data loader to use some hyperparameters, i.e., `CONSTANTS` variables.
After editing the yaml file, one can simply execute

```
python SRC_DIR/main/generate.py generate_model.yaml
```

Again, for convenience, it is recommended to create a soft link for `generate.py`.
