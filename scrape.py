import pandas as pd
from google_image_scraper import GoogleImageScraper

df_actors = pd.read_csv("actors.csv")
actors = df_actors["name"].tolist()
scraper = GoogleImageScraper("images", sleep_between_downloads=0.25)

if __name__ == "__main__":
    scraper.scrape(actors, images_per_query=10, min_face_fraction=0.1)
