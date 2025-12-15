import os
import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    print(f"Downloaded: {save_path}")


base_url = "http://images.cocodataset.org/"
files = {
    # "train2017.zip": "zips/train2017.zip",
    "val2017.zip": "zips/val2017.zip",
    "annotations_trainval2017.zip": "annotations/annotations_trainval2017.zip",
}

save_dir = "../datasets/MS-COCO/"
os.makedirs(save_dir, exist_ok=True)

for filename, path in files.items():
    url = base_url + path
    save_path = os.path.join(save_dir, filename)
    download_file(url, save_path)
