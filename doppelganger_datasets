import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_triplets(
    files: list[str],
    seed: int = 2025,
) -> list[tuple[str, str, str]]:
    actors = sorted(list(set([f.split("__")[0] for f in files])))
    albums = [
        sorted([f for f in files if f.split("__")[0] == actor]) for actor in actors
    ]
    np.random.seed(seed)
    triplets = list()
    for k, album in enumerate(albums):
        negatives_ids = [i for i in range(len(albums)) if i != k]
        for i in range(len(album)):
            for j in range(i + 1, len(album)):
                anchor = album[i]
                positive = album[j]
                negative = str(
                    np.random.choice(albums[np.random.choice(negatives_ids)])
                )
                triplets.append((anchor, positive, negative))
    print(f"Number of photos in dataset = {len(files)}")
    print(f"Number of actors in dataset = {len(actors)}")
    print(f"Number of triplets in dataset = {len(triplets)}")
    return triplets


class TripletDataset(Dataset):
    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        transform,
    ) -> None:
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        path_anchor, path_positive, path_negative = self.triplets[idx]
        anchor = torch.load(path_anchor, weights_only=True)
        positive = torch.load(path_positive, weights_only=True)
        negative = torch.load(path_negative, weights_only=True)
        if self.transform is None:
            return anchor, positive, negative
        else:
            return (
                self.transform(anchor),
                self.transform(positive),
                self.transform(negative),
            )


class ActorDataset(Dataset):
    def __init__(
        self,
        filenames: list[str],
        transform,
    ):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = torch.load(self.filenames[idx], weights_only=True)
        if self.transform is None:
            return image
        else:
            return self.transform(image)
