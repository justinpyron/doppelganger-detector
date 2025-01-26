import logging

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_relevance_matrix(
    files: list[str],
    include_self: bool = False,
) -> np.array:
    relevance_mask = np.array(
        [
            [base.split("__")[0] == other.split("__")[0] for other in files]
            for base in files
        ]
    )
    np.fill_diagonal(relevance_mask, include_self)
    return relevance_mask


def precision_at_k(
    relevance_mask: list[bool],
    relevance_score: list[float],
    k: int,
) -> float:
    predicted_top_k = np.argsort(relevance_score)[-k:]
    precision = relevance_mask[predicted_top_k].mean()
    return precision


def recall_at_k(
    relevance_mask: list[bool],
    relevance_score: list[float],
    k: int,
) -> float:
    predicted_top_k = np.argsort(relevance_score)[-k:]
    recall = relevance_mask[predicted_top_k].sum() / relevance_mask.sum()
    return recall


def mean_precision_at_k(
    relevance_mask: np.array,
    relevance_score: list[float],
    k: int,
):
    return np.array(
        [
            precision_at_k(rel, pred, k)
            for rel, pred in zip(relevance_mask, relevance_score)
        ]
    ).mean()


def mean_recall_at_k(
    relevance_mask: np.array,
    relevance_score: list[float],
    k: int,
):
    return np.array(
        [
            recall_at_k(rel, pred, k)
            for rel, pred in zip(relevance_mask, relevance_score)
        ]
    ).mean()


def get_embedding_vectors(
    image_dataloader: DataLoader,
    model: nn.Module,
) -> torch.Tensor:
    """Do a forward pass to compute embedding vectors"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logits = list()
    with torch.no_grad():
        for i, image in enumerate(image_dataloader):
            image = image.to(device)
            out = model(image)
            if i % 10 == 0:
                logger.info(f"[computing embeddings] {i:4} / {len(image_dataloader)}")
            logits.append(out.cpu())
    return torch.vstack(logits).numpy()


def get_metrics(
    files: list[str],
    actor_loader: DataLoader,
    model: nn.Module,
    k: int,
) -> tuple[float, float]:
    """Compute mean precision @ k and mean recall @ k"""
    logger.info("Evaluating...")
    relevance_mask = make_relevance_matrix(files)
    logits = get_embedding_vectors(actor_loader, model)
    relevance_score = 1 - cdist(logits, logits, metric="cosine")
    np.fill_diagonal(relevance_score, -np.inf)  # Don't count yourself
    precision = mean_precision_at_k(relevance_mask, relevance_score, k)
    recall = mean_recall_at_k(relevance_mask, relevance_score, k)
    return precision, recall
