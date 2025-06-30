import requests
import os

def download_images_from_github(folder_api_url, raw_base_url, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    response = requests.get(folder_api_url)
    if response.status_code != 200:
        print(f"Failed to get data from {folder_api_url}")
        return

    files = response.json()

    for file in files:
        if file['name'].endswith(('.jpg', '.jpeg', '.png')):
            filename = file['name']
            raw_url = f"{raw_base_url}/{filename}"
            print(f"Downloading {filename} ...")
            img_data = requests.get(raw_url).content
            with open(os.path.join(save_dir, filename), 'wb') as f:
                f.write(img_data)

# For "with_mask"
download_images_from_github(
    folder_api_url="https://api.github.com/repos/chandrikadeb7/Face-Mask-Detection/contents/dataset/with_mask",
    raw_base_url="https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/dataset/with_mask",
    save_dir="dataset/with_mask"
)

# For "without_mask"
download_images_from_github(
    folder_api_url="https://api.github.com/repos/chandrikadeb7/Face-Mask-Detection/contents/dataset/without_mask",
    raw_base_url="https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/dataset/without_mask",
    save_dir="dataset/without_mask"
)
