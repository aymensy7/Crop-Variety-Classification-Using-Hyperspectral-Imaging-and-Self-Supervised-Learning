"""
Downloads the Indian Pines hyperspectral dataset (.mat files) into ./data/.

The dataset is hosted publicly by Purdue University's GRSS/AVIRIS archive.
This script tries a direct download; if your network blocks it, the URLs are
printed so you can fetch them manually and place both files in this folder:

    data/Indian_pines_corrected.mat
    data/Indian_pines_gt.mat
"""
import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "Indian_pines_corrected.mat": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    "Indian_pines_gt.mat": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
}


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"✓ {filename} already exists, skipping")
            continue
        print(f"Downloading {filename} from {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✓ Saved to {dest}")
        except Exception as e:
            print(f"✗ Could not download {filename} automatically ({e}).")
            print(f"  Please download it manually from:\n    {url}")
            print(f"  and place it at:\n    {dest}")


if __name__ == "__main__":
    download()
