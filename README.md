# PursuitNet：A deep learning model for predicting pursuit-like behavior in mice
A deep learning framework specifically designed to model the competitive, real-time dynamics of pursuit-escape scenarios.

## License
PursuitNet is licensed under <a rel="license" href="http://www.apache.org/licenses/LICENSE-2.0" 

## Installation
### Install Anaconda
We recommend using Anaconda.
The installation is described on the following page:\
https://docs.anaconda.com/anaconda/install/linux/

### Install Required Packages
```sh
conda env create -f environment.yml
```

### Activate Environment
```sh
conda activate PursuitNet
```

### Install Argoverse API
```sh
pip install git+https://github.com/argoai/argoverse-api.git
```

## Setup Argoverse Dataset
### Download and Extract Dataset
```sh
bash fetch_dataset.sh
```

### Preprocess the Dataset
Online and offline preprocessing is implemented. If you want to train your model offline on the preprocessed dataset, run:
```sh
python3 preprocess.py
```
You can also skip this step and run the preprocessing online during training.
## Train Model
```sh
python3 train.py
```
or
```sh
python3 train.py --use_preprocessed=True
```
Checkpoints are saved in the `lightning_logs/` folder.
For accessing metrics and losses via Tensorboard, first start the server:
```sh
tensorboard --logdir lightning_logs/
```
Navigating to http://localhost:6006/ opens Tensorboard.


## Test Model on Validation Set
```sh
python3 test.py --weight=/path/to/checkpoint.ckpt
```

## Generate Predictions on Test Set
```sh
python3 test.py --weight=/path/to/checkpoint.ckpt --split=test
```
