# Aircrafts classification

Binary classification algorithm to distinguish between NATO and non-NATO military aircrafts, implemented in Pytorch-Lightning.

## Installation
To create a new conda environment with all necessary packages:
- open Anaconda terminal 
- navigate to the repository folder
- type:

```bash
conda env create -f environment.yml
```

## Usage

### Train
In _configs/aircrafts_config.yaml_ insert:
- "image_dir": path to image folder

To start the training procedure: 
```bash
python train.py -c configs/aircrafts_config.yaml
```

### Test
In _configs/aircrafts_config.yaml_ insert:
- "weight_path": path to checkpoint.

To test the model: 
```bash
python test.py -c configs/aircrafts_config.yaml
```

To see performances and confusion metrics:
- run jupyter notebook called EDA.ipynb

## License
[MIT](https://choosealicense.com/licenses/mit/)



