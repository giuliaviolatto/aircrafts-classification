import torch
import pandas as pd
from pathlib import Path
import albumentations as A
from typing import Dict, Any
from torch.utils.data import Dataset

from src.utils import load_rgb


class ClassificationDataset(Dataset):
    def __init__(self,
                 samples: pd.DataFrame,
                 transform: A.Compose,
                 rootdir: Path
                 ) -> None:
        self.samples = samples
        self.transform = transform
        self.rootdir = rootdir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        # Read image
        filename = self.samples.iloc[idx]['filename'] + '.jpg'
        image_path = self.rootdir / filename
        image = load_rgb(image_path)

        # Annotation: 0 or 1
        isnato = self.samples.iloc[idx]['isnato']
        isnato = torch.tensor([isnato]).float()

        # apply augmentations
        sample = self.transform(image=image)
        image = sample["image"]

        return {
            "image": image,
            "label": isnato,
        }