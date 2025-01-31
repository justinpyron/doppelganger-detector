import logging

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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


class ImageDataset(Dataset):
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


def reduce_dim(
    array: np.ndarray,
    n_components: int,
) -> np.ndarray:
    logger.info("Fitting PCA to reduce dimension...")
    pca = PCA(n_components=n_components)
    pca.fit(array)
    array_reduced_dim = pca.transform(array)
    return array_reduced_dim


def create_triplets(
    files: list[str],
    transform,
    seed: int = 2025,
) -> list[tuple[str, str, str]]:
    names = sorted(list(set([f.split("__")[0] for f in files])))
    albums = [
        [i for i, f in enumerate(files) if f.split("__")[0] == name] for name in names
    ]
    image_dataset = ImageDataset(files, transform)
    images = torch.stack(
        [image_dataset[i].flatten() for i in range(len(image_dataset))]
    ).numpy()
    images_reduced_dim = reduce_dim(images, 500)
    distances = cdist(images_reduced_dim, images_reduced_dim)
    triplets = list()
    for album in albums:
        for i, idx_i in enumerate(album):
            idx_most_similar_all = np.argsort(distances[idx_i])[:25]
            # 25 is arbitrary; just needs to be > max album length
            idx_most_similar_other_actors = [
                idx for idx in idx_most_similar_all if idx not in album
            ]
            for j, idx_j in enumerate(album[i + 1 :]):
                anchor = files[idx_i]
                positive = files[idx_j]
                negative = files[idx_most_similar_other_actors[j]]
                triplets.append((anchor, positive, negative))
    logger.info(f"Number of actors in dataset = {len(names)}")
    logger.info(f"Number of photos in dataset = {len(files)}")
    logger.info(f"Number of triplets in dataset = {len(triplets)}")
    return triplets
