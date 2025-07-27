#!/usr/bin/env python3
"""
curate_images.py

WeedScout image curation pipeline for model training.
Performs filtering, deduplication, and feature-based image curation.

Pipeline Overview:
------------------
• Phase 0: Species Filtering
  - Auto-keeps or skips species based on image count and user prompts

• Phase 1: Hand Filtering
  - Uses MobileNetV2-based classifier to detect and remove images with hands or people

• Phase 2: Outlier Filtering
  - Uses EfficientNetV2B0 to extract features
  - Removes outliers via cosine distance

Features:
---------
- Fully commented and clean structure
- Resume support (tracks processed species)
- Side-by-side Rich progress panels
- Minimal backend logging (TensorFlow + cuDNN silenced)
"""

# === Environment Setup (MUST be done before importing tensorflow) ===
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# === Standard Imports ===
import json
import csv
import sys
import logging
from pathlib import Path
from collections import defaultdict

# === TensorFlow + NumPy ===
import tensorflow as tf
import numpy as np
from tensorflow.data import Options
from sklearn.metrics.pairwise import cosine_distances

# === Rich UI ===
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

# === Config ===
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_FOLDER = Path("/mnt/d/training_data")
CURATED_FOLDER = Path("/mnt/d/curated_training_data")

SPECIES_FILE = SCRIPT_DIR / "species.json"
TAXON_CACHE_FILE = SCRIPT_DIR / "gbif_taxon_mappings.json"
OCCURRENCE_FILE = SCRIPT_DIR / "occurrence.txt"
OCCURRENCE_CACHE_FILE = SCRIPT_DIR / "occurrence_counts.json"

PROMPT_CACHE_FILE = SCRIPT_DIR / "prompt_cache.json"
FILTERED_PATHS_FILE = SCRIPT_DIR / "filtered_image_paths.json"
CURATED_SPECIES_FILE = SCRIPT_DIR / "curated_species.json"
CURATION_STATS_FILE = SCRIPT_DIR / "curation_stats.json"
PROCESSED_SPECIES_FILE = SCRIPT_DIR / "processed_species.json"

OCCURRENCE_LINE_COUNT = 12869179
TARGET_IMAGE_COUNT = 350
MINIMUM_IMAGE_COUNT = 300
AUTOKEEP_THRESHOLD = 450
HAND_THRESHOLD = 0.85
BATCH_SIZE = 32
PREFETCH_BATCHES = 6

# === Logger Setup ===
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(console=console, show_path=False)])


logger = logging.getLogger("curate")
csv.field_size_limit(sys.maxsize)

# === Image Resize: Aspect-Fill, Center-Cropped ===
def aspect_fill_resize(img, target_size=(300, 300)):
    original_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    target_h, target_w = target_size
    scale = tf.maximum(target_w / original_shape[1], target_h / original_shape[0])
    new_size = tf.cast(tf.round(original_shape * scale), tf.int32)
    img = tf.image.resize(img, new_size)
    offset_height = (new_size[0] - target_h) // 2
    offset_width = (new_size[1] - target_w) // 2
    return tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_h, target_w)

# === Dataset Builder ===
def image_dataset_from_paths(paths, size, batch_size=BATCH_SIZE, prefetch_batches=PREFETCH_BATCHES):
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])

    def decode(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return aspect_fill_resize(img, size)

    options = Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).with_options(options) \
             .batch(batch_size).prefetch(prefetch_batches)

# === GBIF Occurrence Loader ===
def load_occurrence_counts(taxon_to_species):
    if OCCURRENCE_CACHE_FILE.exists():
        with open(OCCURRENCE_CACHE_FILE) as f:
            return json.load(f)
    counts = defaultdict(int)
    with open(OCCURRENCE_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            taxon_id = row.get("taxonKey")
            if taxon_id and taxon_id.isdigit():
                species = taxon_to_species.get(int(taxon_id))
                if species:
                    counts[species] += 1
    with open(OCCURRENCE_CACHE_FILE, "w") as f:
        json.dump(counts, f, indent=2)
    return counts

# === Load Metadata ===
with open(SPECIES_FILE) as f:
    accepted_species = json.load(f)
with open(TAXON_CACHE_FILE) as f:
    taxon_cache = json.load(f)
    taxon_to_species = {v: k for k, v in taxon_cache.items() if k in accepted_species}
occurrence_counts = load_occurrence_counts(taxon_to_species)

# === Load or Initialize State Files ===
processed_species = set(json.load(open(PROCESSED_SPECIES_FILE))) if PROCESSED_SPECIES_FILE.exists() else set()
filtered_paths = json.load(open(FILTERED_PATHS_FILE)) if FILTERED_PATHS_FILE.exists() else {}
curated_species = json.load(open(CURATED_SPECIES_FILE)) if CURATED_SPECIES_FILE.exists() else []
curation_stats = json.load(open(CURATION_STATS_FILE)) if CURATION_STATS_FILE.exists() else {}
prompt_cache = json.load(open(PROMPT_CACHE_FILE)) if PROMPT_CACHE_FILE.exists() else {}

# === Load Models ===
logger.info("🧠 Loading models...")
hand_model = tf.keras.models.load_model(SCRIPT_DIR / "hand_classifier.keras")
effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, pooling="avg")
preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

# === Phase 0: Decide Which Species to Curate ===
reuse = input("Reuse previous curation answers? [y/n]: ").strip().lower() == "y"
if not reuse:
    prompt_cache = {}

species_to_images = {}
pre_filtered_species = []
skipped = 0

for species in accepted_species:
    folder = SOURCE_IMAGE_FOLDER / species.replace(" ", "_")
    images = list(folder.glob("*.jpg"))
    total_images = len(images)
    occ_count = occurrence_counts.get(species, 0)

    if total_images == 0:
        logger.info(f"❌ {species} has 0 images. Skipping.")
        skipped += 1
        continue
    if total_images >= AUTOKEEP_THRESHOLD:
        keep = True
    elif total_images < MINIMUM_IMAGE_COUNT:
        logger.info(f"⚠️ {species} has only {total_images} images and {occ_count} occurrences. Skipping.")
        skipped += 1
        continue
    elif 300 <= total_images < AUTOKEEP_THRESHOLD:
        keep = prompt_cache.get(species) if reuse else input(f"Keep {species}? ({total_images} images, {occ_count} occurrences) [y/n]: ").strip().lower() == "y"
        prompt_cache[species] = keep
    else:
        keep = False

    if keep:
        pre_filtered_species.append(species)
        species_to_images[species] = images
    else:
        skipped += 1

with open(PROMPT_CACHE_FILE, "w") as f:
    json.dump(prompt_cache, f, indent=2)

logger.info(f"📋 {len(pre_filtered_species)} species selected. {skipped} skipped.")

# === Main Curation Loop ===
total_progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                          "[{task.completed}/{task.total}]", TimeElapsedColumn(), TimeRemainingColumn())
phase_progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                          "[{task.completed}/{task.total}]", TimeElapsedColumn(), TimeRemainingColumn())

total_task = total_progress.add_task("Species curated", total=len(pre_filtered_species))
group = Group(
    total_progress,
    phase_progress
)

with Live(group, console=console, refresh_per_second=2):
    for species in pre_filtered_species:
        if species in processed_species:
            total_progress.update(total_task, advance=1)
            continue

        image_paths = species_to_images[species]
        occ_count = occurrence_counts.get(species, 0)

        # === Phase 1: Hand Filtering ===
        ds_hand = image_dataset_from_paths(image_paths, (224, 224))
        hand_preds = []
        task1 = phase_progress.add_task(f"✋ Hand filtering {species}", total=len(image_paths))
        for batch in ds_hand:
            preds = hand_model.predict(batch, verbose=0)
            hand_preds.extend(preds)
            phase_progress.update(task1, advance=len(preds))
        phase_progress.remove_task(task1)

        kept_paths = [p for p, pred in zip(image_paths, hand_preds) if pred[0] < HAND_THRESHOLD]
        if len(kept_paths) < MINIMUM_IMAGE_COUNT:
            logger.warning(f"⚠️ {species} dropped below threshold after hand filtering ({len(kept_paths)} kept). Skipping.")
            total_progress.update(total_task, advance=1)
            continue
        filtered_paths[species] = [str(p) for p in kept_paths]
        curation_stats[species] = {
            "original_images": len(image_paths),
            "occurrence_count": occ_count,
            "hand_filtered": len(image_paths) - len(kept_paths)
        }

        # === Phase 2: Harmfulness-ranked filtering (Outliers + Duplicates) ===
        ds_feat = image_dataset_from_paths(kept_paths, (300, 300)).map(preprocess_input)
        features = []
        task2 = phase_progress.add_task(f"🧠 Harmfulness filtering {species}", total=len(kept_paths))
        for batch in ds_feat:
            features.append(effnet(batch, training=False).numpy())
            phase_progress.update(task2, advance=len(batch))
        phase_progress.remove_task(task2)

        features = np.vstack(features)
        dist_matrix = cosine_distances(features)

        # Compute outlier and duplicate signals
        mean_dists = np.mean(dist_matrix, axis=1)
        min_dists = np.min(dist_matrix + np.eye(len(features)) * 10, axis=1)  # Ignore self-similarity

        # Combine into a harmfulness score
        # Prioritize the outliers, but still count duplicates with a beta of 0.75
        alpha, beta = 1.0, 0.75
        harmfulness_scores = alpha * mean_dists - beta * min_dists

        if len(features) > TARGET_IMAGE_COUNT:
            removal_count = len(features) - TARGET_IMAGE_COUNT
        else:
            removal_count = max(1, int(len(features) * 0.05))  # Remove at least one image if small set

        worst_idxs = np.argsort(-harmfulness_scores)[:removal_count]
        keep_idxs = [i for i in range(len(features)) if i not in set(worst_idxs)]

        features = features[keep_idxs]
        final_paths = [kept_paths[i] for i in keep_idxs]

        
        # === Save Curated Images ===
        task3 = phase_progress.add_task(f"💾 Saving {species}", total=len(final_paths))
        dest = CURATED_FOLDER / species.replace(" ", "_")
        dest.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(final_paths):
            try:
                img = tf.io.read_file(str(path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = aspect_fill_resize(img, (300, 300))
                img = tf.image.convert_image_dtype(img, tf.uint8)
                tf.io.write_file(str(dest / f"resized_{i:04d}.jpg"), tf.io.encode_jpeg(img))
            except Exception as e:
                logger.warning(f"⚠️ Failed to save image {path.name}: {e}")
            phase_progress.update(task3, advance=1)
        phase_progress.remove_task(task3)

        # === Update Records and Persist ===
        curated_species.append(species)
        processed_species.add(species)
        curation_stats[species]["outlier_removed"] = len(features) - len(final_paths)
        curation_stats[species]["curated_images"] = len(final_paths)

        with open(CURATED_SPECIES_FILE, "w") as f: json.dump(curated_species, f, indent=2)
        with open(FILTERED_PATHS_FILE, "w") as f: json.dump(filtered_paths, f, indent=2)
        with open(CURATION_STATS_FILE, "w") as f: json.dump(curation_stats, f, indent=2)
        with open(PROCESSED_SPECIES_FILE, "w") as f: json.dump(sorted(processed_species), f, indent=2)

        total_progress.update(total_task, advance=1)

logger.info(f"✅ Curation complete. {len(processed_species)} species processed.")
