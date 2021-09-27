import torch
import pytorch_lightning as pl
from src.utils import load_obj


class AircraftClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(AircraftClassifier, self).__init__()
        self.hparams.update(hparams)

        # Model
        self.model = load_obj(self.hparams["model"]["class_name"])(
            pretrained=self.hparams["model"]["params"]["pretrained"]
        )

        # Finetune
        if self.hparams["model"]["params"]["finetune"]:

            # Freeze
            if self.hparams["model"]["params"]["freeze"]:
                for param in self.model.parameters():
                    param.requires_grad = False

            # Substitute FC layer with new one: not frozen
            num_in_features = self.model.fc.in_features  # 2048
            self.model.fc = torch.nn.Linear(num_in_features,
                                            self.hparams["model"]["params"]["classes"])

        # Loss
        self.loss = load_obj(self.hparams["loss"]["class_name"])()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def configure_optimizers(self):

        # Optimizer
        optimizer = load_obj(self.hparams["optimizer"]["class_name"])(
            params=self.parameters(),
            **self.hparams["optimizer"]["params"]
        )

        # Scheduler
        scheduler = load_obj(self.hparams["scheduler"]["class_name"])(
            optimizer=optimizer,
            **self.hparams["scheduler"]["params"]
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        image, label = batch["image"], batch["label"]

        # Output of the Network
        logits = self.forward(image)

        # Loss
        loss = self.loss(logits, label)

        # Log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_id):

        image, label = batch["image"], batch["label"]

        # Output of the Network
        logits = self.forward(image)

        # Loss
        loss = self.loss(logits, label)

        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


