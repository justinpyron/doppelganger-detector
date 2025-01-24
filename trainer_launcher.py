import logging
import os
from datetime import datetime

import numpy as np
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from doppelganger_datasets import ActorDataset, TripletDataset, create_triplets
from trainer import DoppelgangerTrainer

# subset_size = 100
subset_size = 20
train_fraction = 0.9
batch_size_triplets = 32
batch_size_actors = 64
weight_decay = 0.5
margin = 20
k = 9


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

    # Data files
    root = "images/"
    files = sorted([os.path.join(root, f) for f in os.listdir(root)])
    actors = sorted(list(set([f.split("__")[0] for f in files])))

    # Create a subset
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
    resnet_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
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
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

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
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=150)
    trainer.launch_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        margin=margin,
        k=k,
        print_every=10,
    )

    # Epoch 1: Train all weights
    for param in model.parameters():
        param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=150)
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
