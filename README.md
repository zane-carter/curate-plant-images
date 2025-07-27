# Curate Plant Images

Filter and clean large-scale plant image datasets for machine learning. This script removes low-quality, duplicate, and human-obstructed images before resizing them for training.

## Features

- ✋ Detects and removes images with visible hands using a skin-tone threshold
- 📉 Filters low-entropy and outlier images to improve model quality
- 🪞 Deduplicates near-identical images using perceptual hashing
- 📏 Resizes images to a target training resolution (e.g., 224×224)

## Requirements

- Python 3.8+
- TensorFlow
- OpenCV (`cv2`)
- scikit-image
- Pillow
- imagehash
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
