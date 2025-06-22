# loadData.py
# This script handles downloading and preparing the dataset.

import os
import numpy as np
import urllib.request
from urllib.parse import quote
from sklearn.model_selection import train_test_split

DATA_FOLDER = "quickdraw_data"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

def _download_data(categories):
    """
    Checks if .npy files exist for the given categories and downloads them if not.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    print(f"Ensuring data exists in '{DATA_FOLDER}' folder...")
    
    for cat in categories:
        url_cat = quote(cat) # Handle spaces in names
        file_path = os.path.join(DATA_FOLDER, f"{cat}.npy")
        
        if not os.path.exists(file_path):
            print(f"Downloading {cat}.npy...")
            url = f"{BASE_URL}/{url_cat}.npy"
            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                print(f"Failed to download {cat}.npy. Error: {e}")
                # Remove the file if download was incomplete
                if os.path.exists(file_path):
                    os.remove(file_path)

def load_data(categories, sample_size_per_class, test_split=0.2):
    """
    Loads data for the given categories, samples it, and splits it into
    training and testing sets.
    
    Returns:
        (x_train, y_train, x_test, y_test) as NumPy arrays.
    """
    _download_data(categories)
    
    all_images = []
    all_labels = []

    print(f"\nLoading and sampling {sample_size_per_class} images per class...")
    for i, category in enumerate(categories):
        file_path = os.path.join(DATA_FOLDER, f"{category}.npy")
        if not os.path.exists(file_path):
            print(f"Warning: File not found for '{category}'. Skipping.")
            continue
            
        try:
            data = np.load(file_path)
            # Take a random sample from the data for the current class
            # This prevents bias from the original data ordering
            indices = np.random.choice(data.shape[0], sample_size_per_class, replace=False)
            sampled_data = data[indices]
            
            all_images.append(sampled_data)
            all_labels.append(np.full(sample_size_per_class, i))
        except Exception as e:
            print(f"Error loading or sampling '{category}'. Error: {e}")

    # Concatenate all lists into single large numpy arrays
    x = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Pre-process the image data
    # 1. Normalize pixel values from [0, 255] to [0.0, 1.0]
    x = x.astype('float32') / 255.0
    # 2. Reshape for the CNN: (num_samples, height, width, channels)
    x = x.reshape(-1, 28, 28, 1)
    
    print("\nSplitting data into training and testing sets...")
    # Use stratify to ensure all classes are represented proportionally
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_split,
        random_state=42,
        stratify=y
    )
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape:  {x_test.shape}")
    
    return x_train, y_train, x_test, y_test
