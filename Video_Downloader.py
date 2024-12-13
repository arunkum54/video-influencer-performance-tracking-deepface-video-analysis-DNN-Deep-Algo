import os
import pandas as pd
import requests
from tqdm import tqdm

def download_video(url, output_folder="videos"):
    """
    Downloads a video file from a given URL.
    Args:
        url (str): The URL to download the video from.
        output_folder (str): Folder to save the downloaded video.
    Returns:
        str: Path to the downloaded video file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_filename = os.path.join(output_folder, os.path.basename(url))

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(video_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return video_filename
    except requests.RequestException as e:
        print(f"Error downloading video from {url}: {e}")
        return None

# Load video data
csv_file_path = "/Users/arunkumarrana/Desktop/NN/Data.csv"
data = pd.read_csv(csv_file_path)

# Download videos
video_folder = "downloaded_videos"
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    video_url = row['Video URL']
    downloaded_video = download_video(video_url, output_folder=video_folder)

    if downloaded_video:
        print(f"Downloaded video: {downloaded_video}")
    else:
        print(f"Failed to download video from URL: {video_url}")
