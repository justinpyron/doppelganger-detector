import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = "images/"
MODEL_CARD = "trpakov/vit-face-expression"
MODEL_CHECKPOINT = "hf_model_session_2.pt"
FILENAME_EMBEDDINGS = "embeddings.pt"


class Retriever:
    def __init__(self):
        self.files = pd.read_csv("filenames.csv")["filenames"].tolist()
        self.names = np.array([f.split("/")[1].split("__")[0] for f in self.files])
        self.embeddings = torch.load(FILENAME_EMBEDDINGS, weights_only=True)
        self.load_model()

    def load_model(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(MODEL_CARD, use_fast=False)
        embedding_dim = 128
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_CARD)
        head = nn.Linear(self.model.classifier.in_features, embedding_dim)
        self.model.classifier = head
        self.model.load_state_dict(torch.load(MODEL_CHECKPOINT, weights_only=True))
        self.model.eval()

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        img = self.processor(image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            logits = self.model(img).logits
        return logits

    def find(
        self,
        query_image: torch.Tensor,
        k_retrieve: int,
        k_return: int = 1,
    ) -> list[str]:
        embedding = self.embed(query_image)
        distances = cdist(
            embedding,
            self.embeddings,
            metric="cosine",
        )[0]
        idx_top = np.argsort(distances)[1 : k_retrieve + 1]
        names_top = self.names[idx_top]
        return pd.Series(names_top).value_counts().index[:k_return].tolist()
