import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from doppelganger_datasets import ImageDataset, TripletDataset, create_triplets
from trainer import DoppelgangerTrainer

checkpoint = "checkpoint_2025-02-01T16_28_epoch2.pt"
batch_size_triplets = 32
batch_size_actors = 256
lr = 1e-5
start_factor = 0.01
warmup = 500
weight_decay = 1
dropout_prob = 0.1
margin = 0.5
k = 100
embedding_dim = 128
n_epochs = 2
model_card = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_card, use_fast=False)


log_filename = datetime.now().strftime("log_%Y%m%dT%H%M.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Launching train session...")
    logger.info(f"batch_size_triplets = {batch_size_triplets}")
    logger.info(f"batch_size_actors = {batch_size_actors}")
    logger.info(f"weight_decay = {weight_decay}")
    logger.info(f"dropout_prob = {dropout_prob}")
    logger.info(f"lr = {lr}")
    logger.info(f"start_factor = {start_factor}")
    logger.info(f"warmup = {warmup}")
    logger.info(f"embedding_dim = {embedding_dim}")
    logger.info(f"n_epochs = {n_epochs}")

    # Load data
    files_train = pd.read_csv("files_train.csv").filenames.tolist()
    files_val = pd.read_csv("files_val.csv").filenames.tolist()
    df_train = pd.read_csv("triplets_train.csv")
    df_val = pd.read_csv("triplets_val.csv")
    triplets_train = list(df_train.itertuples(index=False, name=None))
    triplets_val = list(df_val.itertuples(index=False, name=None))

    # Triplets datasets
    triplet_dataset_train = TripletDataset(triplets_train, processor)
    triplet_dataset_val = TripletDataset(triplets_val, processor)
    triplet_dataloader_train = DataLoader(
        triplet_dataset_train,
        batch_size=batch_size_triplets,
        num_workers=4,
        shuffle=True,
    )
    triplet_dataloader_val = DataLoader(
        triplet_dataset_val,
        batch_size=batch_size_triplets,
        num_workers=4,
        shuffle=True,
    )

    # Actor datasets
    actor_dataset_train = ImageDataset(files_train, processor)
    actor_dataset_val = ImageDataset(files_val, processor)
    actor_loader_train = DataLoader(
        actor_dataset_train,
        batch_size=batch_size_actors,
        num_workers=4,
        shuffle=False,
    )
    actor_loader_val = DataLoader(
        actor_dataset_val,
        batch_size=batch_size_actors,
        num_workers=4,
        shuffle=False,
    )

    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_card)
    head = nn.Linear(model.classifier.in_features, embedding_dim)
    model.classifier = head
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.config.attention_probs_dropout_prob = dropout_prob
    model.config.hidden_dropout_prob = dropout_prob

    # Create trainer
    trainer = DoppelgangerTrainer(
        files_train=files_train,
        files_val=files_val,
        triplet_dataloader_train=triplet_dataloader_train,
        triplet_dataloader_val=triplet_dataloader_val,
        actor_dataloader_train=actor_loader_train,
        actor_dataloader_val=actor_loader_val,
    )

    # Train all weights
    for name, param in model.named_parameters():
        param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup
    )
    for i in range(n_epochs):
        trainer.launch_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            margin=margin,
            k=k,
            print_every=10,
        )


if __name__ == "__main__":
    main()
