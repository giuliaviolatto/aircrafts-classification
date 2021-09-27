import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import yaml
import torch
import pandas as pd
from pathlib import Path
from pytorch_lightning import seed_everything

from src.utils import get_args, select_device
from src.module import AircraftClassifier
from src.data_module import AircraftsDataModule

"""
Scripts to test the classification model for NATO/non-NATO aircrafts.

Usage:
    python test.py -c configs/aircrafts_configs.yaml 
"""


def main():

    # Input args
    args = get_args()

    # Read parameters
    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    seed_everything(hparams["seed"])

    # Define DataModule
    data = AircraftsDataModule(hparams,
                               hparams["input"]["image_dir"],
                               hparams["input"]["labels_test_csv"],
                               'test')
    data.setup()

    # Set device CUDA
    device = select_device('')
    half = device.type != 'cpu'

    # Model
    model = AircraftClassifier.load_from_checkpoint(hparams["test"]['weight_path'],
                                                    hparams=hparams,
                                                    map_location=device)

    # Send model to current device
    model.cuda()
    if next(model.parameters()).is_cuda:
        print('Model loaded successfully to Device\n')

    # if half:
    #     model.half()  # to FP16

    true_labels = []
    pred_labels = []

    for batch in data.test_dataloader():

        image_batch = batch["image"].to(device)
        true_label_batch = batch["label"].tolist()
        true_label_batch = [label for label_sublist in true_label_batch for label in label_sublist]
        true_label_batch = [int(i) for i in true_label_batch]

        # Store true label
        true_labels.extend(true_label_batch)

        # Output of the model
        logits = model(image_batch)
        probs = torch.sigmoid(logits)

        # Threshold the probability
        pred_label_batch = (probs >= 0.5).int().tolist()
        pred_label_batch = [pred_label for label_sublist in pred_label_batch for pred_label in label_sublist]

        pred_labels.extend(pred_label_batch)

    # Create table with true and pred labels
    df = pd.DataFrame({"True_labels": true_labels,
                       "Pred_labels": pred_labels
                       })

    # Save results
    result_folder = Path(hparams["test"]["result_folder"])
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    df.to_csv(result_folder / hparams["test"]["file_name"], index=False)
    print('Result file saved.')


if __name__ == '__main__':
    main()