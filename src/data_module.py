import pandas as pd
from pathlib import Path
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from src.sampler import AircraftsSampler
from src.dataset import ClassificationDataset


class AircraftsDataModule(pl.LightningDataModule):
    def __init__(self, hparams, image_path, annotations_path, mode):
        """
        Overwrite init method.
        :param hparams: dictionary with hyper-parameters
        :param image_path: path to folder containing images
        :param annotations_path: path to csv files containing training labels.
        :param mode: train or test.
        """
        super().__init__()

        # Hyper-parameters and paths
        self.hparams = hparams
        self.image_path = Path(image_path)
        self.annotations_path = Path(annotations_path)
        self.mode = mode

        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=0):

        # Dataset Augmentation
        transforms_train = A.Compose([A.Resize(height=self.hparams["data"]["img_size"],
                                               width=self.hparams["data"]["img_size"],
                                               always_apply=True),
                                      A.VerticalFlip(),
                                      A.HorizontalFlip(),
                                      A.Rotate(limit=10),
                                      A.ShiftScaleRotate(),
                                      A.Normalize(mean=tuple(self.hparams["data"]["mean"]),
                                                  std=tuple(self.hparams["data"]["std"])),
                                      ToTensorV2()])

        transforms_val = A.Compose([A.Resize(height=self.hparams["data"]["img_size"],
                                             width=self.hparams["data"]["img_size"],
                                             always_apply=True),
                                    A.Normalize(mean=tuple(self.hparams["data"]["mean"]),
                                                std=tuple(self.hparams["data"]["std"])),
                                    ToTensorV2()])

        transforms_test = A.Compose([A.Resize(height=self.hparams["data"]["img_size"],
                                              width=self.hparams["data"]["img_size"],
                                              always_apply=True),
                                     A.Normalize(mean=tuple(self.hparams["data"]["mean"]),
                                                 std=tuple(self.hparams["data"]["std"])),
                                     ToTensorV2()])

        # Read annotation file
        annotation_df = pd.read_csv(self.annotations_path)

        if self.mode == 'train':
            # Train/val split: k-fold
            skf = StratifiedKFold(n_splits=self.hparams["data"]["k_fold"],
                                  shuffle=True,
                                  random_state=self.hparams["seed"])

            for ind, (train_index, val_index) in enumerate(skf.split(annotation_df["filename"], annotation_df["isnato"])):

                # For limited resources, choose one split only
                if ind == self.hparams["data"]["split"]:

                    # Sliced dataframes
                    self.train_samples = annotation_df.iloc[train_index, :].reset_index(drop=True)
                    self.val_samples = annotation_df.iloc[val_index, :].reset_index(drop=True)

            # Define datasets
            self.train_dataset = ClassificationDataset(self.train_samples,
                                                       transforms_train,
                                                       self.image_path)

            self.val_dataset = ClassificationDataset(self.val_samples,
                                                     transforms_val,
                                                     self.image_path)

            print('\nLen train samples = ', len(self.train_samples))
            print('Len val samples = ', len(self.val_samples))

        else:
            self.test_samples = annotation_df
            self.test_dataset = ClassificationDataset(self.test_samples,
                                                      transforms_test,
                                                      self.image_path)
            print('Len test samples = ', len(self.test_samples))

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          sampler=AircraftsSampler(self.train_samples),
                          batch_size=self.hparams["data"]["train_loader"]["batch_size"],
                          num_workers=self.hparams["data"]["train_loader"]["num_workers"],
                          shuffle=self.hparams["data"]["train_loader"]["shuffle"],
                          pin_memory=self.hparams["data"]["train_loader"]["pin_memory"]
                          )

    def val_dataloader(self):

        return DataLoader(self.val_dataset,
                          batch_size=self.hparams["data"]["val_loader"]["batch_size"],
                          num_workers=self.hparams["data"]["val_loader"]["num_workers"],
                          shuffle=self.hparams["data"]["val_loader"]["shuffle"],
                          pin_memory=self.hparams["data"]["val_loader"]["pin_memory"]
                          )

    def test_dataloader(self):

        return DataLoader(self.test_dataset,
                          batch_size=self.hparams["data"]["test_loader"]["batch_size"],
                          num_workers=self.hparams["data"]["test_loader"]["num_workers"],
                          shuffle=self.hparams["data"]["test_loader"]["shuffle"],
                          pin_memory=self.hparams["data"]["test_loader"]["pin_memory"]
                          )