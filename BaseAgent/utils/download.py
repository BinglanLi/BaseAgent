"""Download utilities: S3 file sync and zip extraction."""

import os
import tempfile
import zipfile
from urllib.parse import urljoin

import requests
import tqdm


def check_or_create_path(path=None):
    # Set a default path if none is provided
    if path is None:
        path = os.path.join(os.getcwd(), "tmp_directory")

    # Check if the path exists
    if not os.path.exists(path):
        # If it doesn't exist, create the directory
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")

    return path


def download_and_unzip(url: str, dest_dir: str) -> str:
    """Download a zip file from a URL and extract it to the destination directory.

    Args:
        url: The URL to download the zip file from.
        dest_dir: The directory to extract the contents to.

    Returns:
        The path to the extracted directory, or an error message.

    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading from {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                with tqdm.tqdm(
                    total=total_size / (1024**3),
                    unit="GB",
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            tmp_file.write(chunk)
                            pbar.update(len(chunk) / (1024**3))
                tmp_zip_path = tmp_file.name
        print(f"Downloaded to {tmp_zip_path}. Extracting...")
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        os.unlink(tmp_zip_path)
        print(f"Extraction complete to {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"Error downloading or extracting zip: {e}")
        return f"Error: {e}"


def check_and_download_s3_files(
    s3_bucket_url: str, local_data_lake_path: str, expected_files: list[str], folder: str = "data_lake"
) -> dict[str, bool]:
    """Check for missing files in the local data lake and download them from S3 bucket.

    Args:
        s3_bucket_url: Base URL of the S3 bucket (e.g., "https://biomni-release.s3.amazonaws.com")
        local_data_lake_path: Local path to the data lake directory
        expected_files: List of expected file names in the data lake
        folder: S3 folder name ("data_lake" or "benchmark")

    Returns:
        Dictionary mapping file names to download success status
    """

    os.makedirs(local_data_lake_path, exist_ok=True)
    download_results = {}

    def download_with_progress(url: str, file_path: str, desc: str) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                if total_size > 0:
                    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=desc, ncols=80) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"✗ Failed to download {desc}: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return False

    def cleanup_file(file_path: str):
        """Clean up file if it exists."""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    # Handle benchmark folder (download as zip)
    if folder == "benchmark":
        print(f"Downloading entire {folder} folder structure...")
        s3_zip_url = urljoin(s3_bucket_url + "/", folder + ".zip")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name

            if download_with_progress(s3_zip_url, tmp_zip_path, f"{folder}.zip"):
                print(f"Extracting {folder}.zip...")
                try:
                    with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(local_data_lake_path)
                    print(f"✓ Successfully downloaded and extracted {folder} folder")
                    download_results = dict.fromkeys(expected_files, True)
                except Exception as e:
                    print(f"✗ Error extracting {folder}.zip: {e}")
                    download_results = dict.fromkeys(expected_files, False)
                finally:
                    cleanup_file(tmp_zip_path)
            else:
                download_results = dict.fromkeys(expected_files, False)

        return download_results

    # Handle data_lake folder (download individual files)
    for filename in expected_files:
        local_file_path = os.path.join(local_data_lake_path, filename)

        if os.path.exists(local_file_path):
            download_results[filename] = True
            continue

        s3_file_url = urljoin(s3_bucket_url + "/" + folder + "/", filename)
        print(f"Downloading {filename} from {folder}...")

        if download_with_progress(s3_file_url, local_file_path, filename):
            print(f"✓ Successfully downloaded: {filename}")
            download_results[filename] = True
        else:
            download_results[filename] = False

    return download_results
