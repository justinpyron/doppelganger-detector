import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from doppelganger_datasets import ImageDataset, TripletDataset, create_triplets
from trainer import DoppelgangerTrainer

subset_size = 1000
train_fraction = 0.9
batch_size_triplets = 16
batch_size_actors = 32
lr = 1e-3
start_factor = 0.01
warmup = 1000
weight_decay = 0.5
margin = 1
k = 9
embedding_dim = 256
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
    logger.info(f"lr = {lr}")
    logger.info(f"start_factor = {start_factor}")
    logger.info(f"warmup = {warmup}")

    # Data files
    root = "images/"
    files = sorted([os.path.join(root, f) for f in os.listdir(root)])
    actors = sorted(list(set([f.split("__")[0] for f in files])))

    # Create a subset
    np.random.seed(2025)
    actors_subset = np.random.permutation(actors)[:subset_size]
    idx = np.random.permutation(np.arange(subset_size))
    idx_train = idx[: int(subset_size * train_fraction)]
    idx_val = idx[int(subset_size * train_fraction) :]
    actors_train = [actors_subset[i] for i in idx_train]
    actors_val = [actors_subset[i] for i in idx_val]
    files_train = sorted(
        [f for actor in actors_train for f in files if f.split("__")[0] == actor]
    )
    files_val = sorted(
        [f for actor in actors_val for f in files if f.split("__")[0] == actor]
    )

    # Triplets datasets
    triplets_train = create_triplets(files_train, processor, n_components=10)
    triplets_val = create_triplets(files_val, processor, n_components=10)
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

    # Load pre-trained model
    model = AutoModelForImageClassification.from_pretrained(model_card)
    head = nn.Linear(model.classifier.in_features, embedding_dim)
    model.classifier = head

    # Create trainer
    trainer = DoppelgangerTrainer(
        files_train=files_train,
        files_val=files_val,
        triplet_dataloader_train=triplet_dataloader_train,
        triplet_dataloader_val=triplet_dataloader_val,
        actor_dataloader_train=actor_loader_train,
        actor_dataloader_val=actor_loader_val,
    )

    # Epoch 0: Freeze all weights except last fully-connected layer
    for name, param in model.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup
    )
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
