import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils import get_args
from src.module import AircraftClassifier
from src.data_module import AircraftsDataModule


"""
Scripts to start training of classification model for NATO/non-NATO aircrafts.

Usage:
    python train.py -c configs/aircrafts_configs.yaml 
"""


def main():

    # Input args
    args = get_args()

    # Read parameters
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    # For reproducibility
    seed_everything(hparams["seed"])

    # Logger
    model_architecture = hparams["model"]['class_name'].split(".", 1)[-1]
    logger = TensorBoardLogger(save_dir=hparams["logger"]["project"],
                               name=model_architecture)

    # Define DataModule
    dataset = AircraftsDataModule(hparams,
                                  hparams["input"]["image_dir"],
                                  hparams["input"]["labels_train_csv"],
                                  'train')
    dataset.setup()

    # Init Model
    model = AircraftClassifier(hparams)

    # Checkpoint callback - save best one based on val
    checkpoint_callback = ModelCheckpoint(**hparams["callbacks"]["checkpoint"])
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # Init trainer
    trainer = Trainer(**hparams["trainer"],
                      logger=logger,
                      callbacks=[checkpoint_callback, lr_monitor_callback])

    trainer.fit(model=model, datamodule=dataset)


if __name__ == '__main__':
    main()