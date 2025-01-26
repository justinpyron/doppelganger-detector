import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18, ResNet50_Weights, resnet50

from doppelganger_datasets import ActorDataset, TripletDataset, create_triplets
from trainer import DoppelgangerTrainer

subset_size = 1000
train_fraction = 0.9
batch_size_triplets = 32
batch_size_actors = 64
lr_head_only = 1e-4
lr = 1e-4
start_factor_head_only = 0.01
start_factor = 0.01
weight_decay = 0.5
margin = 40
k = 9
model_type = resnet18
model_weights = ResNet18_Weights.IMAGENET1K_V1
# model_type = resnet50
# model_weights = ResNet50_Weights.IMAGENET1K_V2


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
    logger.info(f"lr_head_only = {lr_head_only}")
    logger.info(f"lr = {lr}")
    logger.info(f"start_factor_head_only = {start_factor_head_only}")
    logger.info(f"start_factor = {start_factor}")

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
    resnet_transform = model_weights.transforms()
    triplets_train = create_triplets(files_train)
    triplets_val = create_triplets(files_val)
    triplet_dataset_train = TripletDataset(triplets_train, transform=resnet_transform)
    triplet_dataset_val = TripletDataset(triplets_val, transform=resnet_transform)
    triplet_dataloader_train = DataLoader(
        triplet_dataset_train,
        batch_size=batch_size_triplets,
        num_workers=4,
        shuffle=True,
    )
    triplet_dataloader_val = DataLoader(
        triplet_dataset_val, batch_size=batch_size_triplets, num_workers=4, shuffle=True
    )

    # Actor datasets
    actor_dataset_train = ActorDataset(files_train, resnet_transform)
    actor_dataset_val = ActorDataset(files_val, resnet_transform)
    actor_loader_train = DataLoader(
        actor_dataset_train, batch_size=batch_size_actors, num_workers=4, shuffle=False
    )
    actor_loader_val = DataLoader(
        actor_dataset_val, batch_size=batch_size_actors, num_workers=4, shuffle=False
    )

    # Load pre-trained model
    model = model_type(weights=model_weights)
    # Load model you've trained already
    # model = model_type()
    # model.load_state_dict(torch.load("checkpoint_2025-01-24T16_23_epoch3.pt", weights_only=True))

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
        if not "fc" in name:
            param.requires_grad = False
    # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=weight_decay)
    optimizer = AdamW(model.parameters(), lr=lr_head_only, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor_head_only, total_iters=500)
    trainer.launch_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        margin=margin,
        k=k,
        print_every=10,
    )

    # Epoch 1+: Train all weights
    for param in model.parameters():
        param.requires_grad = True
    # optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=weight_decay)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=500)
    for i in range(10):
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
