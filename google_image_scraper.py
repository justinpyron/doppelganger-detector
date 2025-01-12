import os
import numpy as np
import torch
import time
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from image_judge import ImageJudge


GOOGLE_IMAGE_SEARCH_URL = "https://www.google.com/search?udm=2&tbm=isch&tbs=itp:face"
# Restrict searches to faces with the `&tbm=isch&tbs=itp:face` parameters

class GoogleImageScraper:
    
    def __init__(
        self,
        destination_folder: str,
        sleep_between_downloads: float,
    ) -> None:
        self.destination_folder = destination_folder
        self.sleep_between_downloads = sleep_between_downloads
        self.image_judge = ImageJudge()

    def fetch_html(
        self,
        query: str,
    ) -> BeautifulSoup:
        """Returns HTML of a Google Images search result page"""
        query = query.replace(" ", "+")
        url = f"{GOOGLE_IMAGE_SEARCH_URL}&q={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup

    def fetch_image_urls(
        self,
        soup: BeautifulSoup,
    ) -> list[str]:
        """Returns list of urls of images in a Google Images search result page"""
        image_elements = soup.find_all("img")
        image_urls = [img.get("src") for img in image_elements if img.get("alt")==""]
        return image_urls

    def fetch_image_from_url(self, url: str) -> torch.tensor:
        """Returns tensor of image at url"""
        try:
            response = requests.get(url)
            with BytesIO(response.content) as f:
                with Image.open(f) as image:
                    return np.array(image)
        except:
            pass

    def fetch_images(
        self,
        query: str,
        images_per_query: int,
        min_face_fraction: float,
    ) -> list[torch.tensor]:
        """Returns list of tensors of images in a Google Images search result page"""
        html = self.fetch_html(query)
        image_urls = self.fetch_image_urls(html)
        images = list()
        for url in image_urls:
            image = self.fetch_image_from_url(url)
            time.sleep(self.sleep_between_downloads)
            if self.image_judge.image_is_acceptable(image, min_face_fraction):
                images.append(image)
            if len(images) >= images_per_query:
                break
        return images

    def scrape(
        self,
        queries: list[str],
        images_per_query: int,
        min_face_fraction: float,
    ) -> None:
        """
        For each query in a list of queries, scrapes then saves
        the images in a Google Images search result page
        """
        N = len(queries)
        for i, query in enumerate(queries):
            try:
                images = self.fetch_images(query, images_per_query, min_face_fraction)
                for j, image in enumerate(images):
                    image_channel_last_dim = torch.permute(torch.tensor(image), (2,0,1))
                    query_cleaned = query.lower().replace(" ", "_")
                    filename = os.path.join(
                        self.destination_folder,
                        f"{query_cleaned}__{j:02}.pt",
                    )
                    torch.save(image_channel_last_dim, filename)
                print(f"{query:30} [{i+1:4}/{N}] Number of images = {len(images):2}")
            except:
                pass
