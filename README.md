# PursuitNet：A deep learning model for predicting pursuit-like behavior in mice
A deep learning framework specifically designed to model the competitive, real-time dynamics of pursuit-escape scenarios.

![image](https://github.com/user-attachments/assets/e920de59-c297-43c1-9edf-4238b2fa9cfb)

# Pursuit-Escape Confrontation (PEC) dataset
The PEC dataset was derived from real animal behaviors observed during predator-prey interactions between a hungry mouse 
and an escaping robotic bait.

![image](https://github.com/user-attachments/assets/f9a04b5c-0007-447a-b527-da3b98b39ea5)

The constituent dataset (*.csv files, 1169) contain the following parameters:
TIMESTAMP: The timestamp indicates the specific time at which the data was recorded. In this example, the timestamp is measured in seconds.
TRACK_ID: The tracking ID is used to uniquely identify each tracked subject (robotic bait or mice). Here, a UUID (Universally Unique Identifier) is used. 
OBJECT_TYPE: The object type specifies whether the recorded data belongs to the robotic bait (BAIT) or a mouse (MICE).
X: The X coordinate represents the position of the subject in a two-dimensional space, measured in pixel.
Y: The Y coordinate, together with the X coordinate, determines the subject's position in two-dimensional space.
SPEED: Velocity indicates the rate of movement of the subject at the recorded timestamp, measured in centimeters per second.
ACCELERATION: Acceleration represents the rate of change of the subject's speed. The unit is typically centimeters per second squared.
SOURCE: The data source indicates from which experimental batch or recording device the data was obtained. This helps in tracking the origin of the data and managing it.


## License
PursuitNet is licensed under
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
