import os
import json
import requests
from pathlib import Path
import ipdb

ROOT = Path(__file__).resolve().parent.parent  # project_root
DATASETS = ROOT / "datasets"

json_file = DATASETS / "NOCAPS/nocaps_val_4500_captions.json"
output_dir = DATASETS / "NOCAPS/nocaps_validation_images"

os.makedirs(output_dir, exist_ok=True)

with open(json_file, "r") as f:
    data = json.load(f)


image_caption_mapping = {}   
failed_downloads = []   
ipdb.set_trace()

for item in data["images"]:
    image_id = item["open_images_id"]  
    image_url = item["coco_url"]  

    image_path = os.path.join(output_dir, f"{image_id}.jpg")

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(image_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {image_id}.jpg")
        else:
            print(f"Failed to download: {image_id}")
            failed_downloads.append(image_id)
    except Exception as e:
        print(f"Error downloading {image_id}: {e}")
        failed_downloads.append(image_id)


if failed_downloads:
    with open("failed_downloads.txt", "w") as f:
        f.write("\n".join(failed_downloads))

print("Image download and caption mapping complete!")