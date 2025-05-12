import math
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl

__all__ = [
    "GazeDataset",
    "GazeDataModule",
    "SingleModel",
    "EyesModel",
    "FullModel",
]

def _identity(img):
    return img

class GazeDataset(Dataset):

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        img_types: Sequence[str] = ("l_eye", "r_eye"),
        augment: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"{self.data_dir} does not exist")

        df = pd.read_csv(self.data_dir / "positions.csv")
        df["filename"] = df["id"].astype(str) + ".jpg"

        self.img_types = list(img_types)
        self._files = df["filename"].tolist()
        self._targets = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
        self._head_angle = torch.tensor(df["head_angle"].values, dtype=torch.float32)

        jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
        # self._transform = transforms.Compose([jitter if augment else transforms.Lambda(lambda x: x), transforms.ToTensor()]) old one
        self._transform = transforms.Compose([jitter if augment else transforms.Lambda(_identity), transforms.ToTensor()])
        self._to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int):
        sample = {"targets": self._targets[idx]}

        if "head_angle" in self.img_types:
            sample["head_angle"] = self._head_angle[idx]

        for t in self.img_types:
            if t == "head_angle":
                continue

            img_path = self.data_dir / t / self._files[idx]
            img = Image.open(img_path)

            # Head‑pos mask is already 1‑channel B&W – no colour jitter.
            tensor = self._to_tensor(img) if t == "head_pos" else self._transform(img)
            sample[t] = tensor

        return sample


class GazeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        batch_size: int = 128,
        num_workers: int = 8,
        train_prop: float = 0.8,
        val_prop: float = 0.1,
        img_types: Sequence[str] = ("l_eye", "r_eye", "head_angle"),
        seed: int = 87,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)


    def setup(self, stage: str | None = None):
        dataset = GazeDataset(self.hparams.data_dir, self.hparams.img_types, augment = (stage == "fit"))
        n_total = len(dataset)
        n_train = int(n_total * self.hparams.train_prop)
        n_val = int(n_total * self.hparams.val_prop)
        n_test = n_total - n_train - n_val

        self.ds_train, self.ds_val, self.ds_test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


def _conv_block(in_ch: int, out_ch: int, ks: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvStack(nn.Sequential):
    def __init__(self, in_ch: int, channels: Sequence[int] = (32, 64, 128), ks: int = 3):
        layers = []
        c_curr = in_ch
        for c_next in channels:
            layers.append(_conv_block(c_curr, c_next, ks))
            c_curr = c_next
        layers.append(nn.AdaptiveAvgPool2d(1))  # (B, c_curr, 1, 1)
        super().__init__(*layers)
        self.out_channels = c_curr

class _Base(pl.LightningModule):
    def __init__(self, lr: float = 3e-4):
        super().__init__()
        self.lr = lr
        self.criterion = nn.SmoothL1Loss()
        self.mae = torchmetrics.MeanAbsoluteError()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return opt

    def _shared_step(self, preds: torch.Tensor, targets: torch.Tensor, stage: str):
        loss = self.criterion(preds, targets)
        mae = self.mae(preds, targets)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_epoch=True, on_step=True)

        return loss


class SingleModel(_Base):
    def __init__(
        self,
        img_type: str = "face_aligned",
        lr: float = 3e-4,
        channels: Sequence[int] = (32, 64, 128),
        hidden: int = 256,
    ):
        super().__init__(lr)
        self.img_type = img_type
        self.backbone = ConvStack(3, channels)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.example_input_array = torch.rand(1, 3, 64, 64)


    def forward(self, x):
        return self.regressor(self.backbone(x))

    def training_step(self, batch, batch_idx):
        preds = self(batch[self.img_type])
        return self._shared_step(preds, batch["targets"], "train")

    def validation_step(self, batch, batch_idx):
        preds = self(batch[self.img_type])
        self._shared_step(preds, batch["targets"], "val")

    def test_step(self, batch, batch_idx):
        preds = self(batch[self.img_type])
        self._shared_step(preds, batch["targets"], "test")


class EyesModel(_Base):
    def __init__(
        self,
        lr: float = 3e-4,
        channels: Sequence[int] = (32, 64, 128),
        hidden: int = 256,
    ):
        super().__init__(lr)
        self.l_stack = ConvStack(3, channels)
        self.r_stack = ConvStack(3, channels)
        out_dim = self.l_stack.out_channels * 2
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.example_input_array = [torch.rand(1, 3, 64, 64)] * 2

    def forward(self, l_eye, r_eye):
        l = self.l_stack(l_eye).flatten(start_dim=1)
        r = self.r_stack(r_eye).flatten(start_dim=1)
        return self.regressor(torch.cat([l, r], dim=1))

    def training_step(self, batch, batch_idx):
        preds = self(batch["l_eye"], batch["r_eye"])
        return self._shared_step(preds, batch["targets"], "train")

    def validation_step(self, batch, batch_idx):
        preds = self(batch["l_eye"], batch["r_eye"])
        self._shared_step(preds, batch["targets"], "val")

    def test_step(self, batch, batch_idx):
        preds = self(batch["l_eye"], batch["r_eye"])
        self._shared_step(preds, batch["targets"], "test")


class FullModel(_Base):
    def __init__(
        self,
        lr: float = 3e-4,
        face_channels: Sequence[int] = (32, 64, 128),
        eye_channels: Sequence[int] = (32, 64, 128),
        head_pos_channels: Sequence[int] = (16, 32, 64),
        hidden: int = 256,
    ):
        super().__init__(lr)
        self.face_stack = ConvStack(3, face_channels)
        self.l_stack = ConvStack(3, eye_channels)
        self.r_stack = ConvStack(3, eye_channels)
        self.head_stack = ConvStack(1, head_pos_channels)

        fused_dim = (
            self.face_stack.out_channels
            + self.l_stack.out_channels * 2
            + self.head_stack.out_channels
            + 1  # head angle scalar
        )
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.example_input_array = [torch.rand(1, 3, 64, 64)] * 3 + [
            torch.rand(1, 1, 64, 64),
            torch.rand(1),
        ]

    def forward(self, face, l_eye, r_eye, head_pos, head_angle):
        face_f = self.face_stack(face).flatten(start_dim=1)
        l_f = self.l_stack(l_eye).flatten(start_dim=1)
        r_f = self.r_stack(r_eye).flatten(start_dim=1)
        h_f = self.head_stack(head_pos).flatten(start_dim=1)
        x = torch.cat([face_f, l_f, r_f, h_f, head_angle.unsqueeze(1)], dim=1)
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        preds = self(
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        return self._shared_step(preds, batch["targets"], "train")

    def validation_step(self, batch, batch_idx):
        preds = self(
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        self._shared_step(preds, batch["targets"], "val")

    def test_step(self, batch, batch_idx):
        preds = self(
            batch["face_aligned"],
            batch["l_eye"],
            batch["r_eye"],
            batch["head_pos"],
            batch["head_angle"],
        )
        self._shared_step(preds, batch["targets"], "test")
