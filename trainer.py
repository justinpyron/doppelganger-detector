import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from metrics import get_metrics

logger = logging.getLogger(__name__)


class DoppelgangerTrainer:
    def __init__(
        self,
        files_train: list[str],
        files_val: list[str],
        triplet_dataloader_train: DataLoader,
        triplet_dataloader_val: DataLoader,
        actor_dataloader_train: DataLoader,
        actor_dataloader_val: DataLoader,
    ) -> None:
        self.files_train = files_train
        self.files_val = files_val
        self.triplet_dataloader_train = triplet_dataloader_train
        self.triplet_dataloader_val = triplet_dataloader_val
        self.actor_dataloader_train = actor_dataloader_train
        self.actor_dataloader_val = actor_dataloader_val
        self.loss_train = list()
        self.loss_val = list()
        self.n_epochs = 0
        self.birthday = datetime.now().strftime("%Y-%m-%dT%H_%M")

    def launch_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        margin: float,
        k: int = 10,
        print_every: int = 10,
    ) -> None:
        checkpoint_name = f"checkpoint_{self.birthday}_epoch{self.n_epochs}.pt"
        logger.info("-" * 80)
        logger.info(f"Epoch {self.n_epochs}")
        logger.info(f"Checkpoint name = {checkpoint_name}")
        logger.info("ARGUMENTS")
        logger.info(f"margin = {margin}")
        logger.info(f"k = {k}")
        start = time.time()

        # DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # TRAIN SET
        logger.info("TRAIN")
        l_train = list()
        model.train()
        for i, (anchor, positive, negative) in enumerate(self.triplet_dataloader_train):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            out_anchor = model(anchor)
            out_positive = model(positive)
            out_negative = model(negative)
            loss = F.triplet_margin_loss(
                out_anchor, out_positive, out_negative, margin=margin
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            l_train.append(loss.item())
            if i % print_every == 0:
                logger.info(
                    f"Batch {i:4} / {len(self.triplet_dataloader_train)} - "
                    f"Loss {loss.item():5.2f} - "
                    f"Time (min) {(time.time() - start)/60:4.1f}"
                )
        self.loss_train.append(l_train)

        # VAL SET
        logger.info("VAL")
        l_val = list()
        model.eval()
        with torch.no_grad():
            for i, (anchor, positive, negative) in enumerate(
                self.triplet_dataloader_val
            ):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                out_anchor = model(anchor)
                out_positive = model(positive)
                out_negative = model(negative)
                loss = F.triplet_margin_loss(
                    out_anchor, out_positive, out_negative, margin=margin
                )
                l_val.append(loss.item())
                if i % print_every == 0:
                    logger.info(
                        f"Batch {i:4} / {len(self.triplet_dataloader_val)} - "
                        f"Loss {loss.item():5.2f} - "
                        f"Time (min) {(time.time() - start)/60:4.1f}"
                    )
        self.loss_val.append(l_val)

        # METRICS
        ma_size = 10
        logger.info(f"Train loss avg = {np.array(l_train).mean()}")
        logger.info(f"Val   loss avg = {np.array(l_val).mean()}")
        logger.info(f"Train loss time series ({ma_size} moving avg)")
        logger.info(pd.Series(l_train).rolling(ma_size).mean().round(3).tolist())
        logger.info(f"Val loss time series ({ma_size} moving avg)")
        logger.info(pd.Series(l_val).rolling(ma_size).mean().round(3).tolist())
        precision_train, recall_train = get_metrics(
            self.files_train, self.actor_dataloader_train, model, k=k
        )
        precision_val, recall_val = get_metrics(
            self.files_val, self.actor_dataloader_val, model, k=k
        )
        logger.info(f"precision_train = {precision_train:.3f}")
        logger.info(f"recall_train    = {recall_train:.3f}")
        logger.info(f"precision_val   = {precision_val:.3f}")
        logger.info(f"recall_val      = {recall_val:.3f}")
        self.n_epochs += 1

        # SAVE
        checkpoint = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(checkpoint, checkpoint_name)
