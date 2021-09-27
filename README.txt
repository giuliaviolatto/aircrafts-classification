Giulia Violatto
24/09/2021


To install the environment:
- open anaconda terminal 
- navigate to this folder
- type: conda env create -f environment.yml

In configs/aircrafts_config.yaml insert:
- path to image folder (1536 images), in variable "image_dir"
- path to checkpoint, in variable weight_path.

Download the model weights from the link below:
https://drive.google.com/drive/folders/1g30xj77MCF9s8QNWM_225xww2J-DG0cY?usp=sharing


To start training:
python train.py -c configs/aircrafts_config.yaml


To Test:
python test.py -c configs/aircrafts_config.yaml


To see performances and confusion metrics:
run jupyter notebook called EDA.ipynb