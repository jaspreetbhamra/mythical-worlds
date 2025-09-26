# src/ingestion/download_gutenberg.py
import requests
import os


def download_gutenberg_text(gutenberg_id: int, title: str, save_dir="data/raw/"):
    """
    Download a text from Project Gutenberg given its ebook ID.
    Saves as a .txt file in save_dir.

    Args:
        gutenberg_id (int): The Project Gutenberg ebook ID (e.g., 6130 for Iliad).
        title (str): A short title/identifier for the file.
        save_dir (str): Directory to save the file.
    """
    # Gutenberg plain text URL (UTF-8 encoding)
    url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url}, status: {response.status_code}")

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{title}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"✅ Saved '{title}' to {file_path}")


if __name__ == "__main__":
    # Example downloads (you can extend this list)
    texts_to_download = {
        "iliad": 6130,
        "odyssey": 1727,
        "beowulf": 16328,
        "le_morte_darthur": 1251,
        "arabian_nights_vol1": 5667,
    }

    for title, gid in texts_to_download.items():
        try:
            download_gutenberg_text(gid, title)
        except Exception as e:
            print(f"❌ Failed for {title}: {e}")
