# Asteroid_detection_CNN

Convolutional neural networks for detecting faint asteroid trails in wide-field astronomical images, developed in the context of Rubin Observatory / LSST data analysis.

This repository contains a **PyTorch-based production implementation** for pixel-level detection of trailed moving objects, together with tools for dataset generation, training, evaluation, and post-processing.

---

## Scientific context

Detecting faint asteroid trails in modern sky surveys is challenging due to:
- low surface brightness of trails,
- variable observing conditions,
- strong background contamination,
- limitations of classical detection pipelines.

This project explores **deep-learning-based pixelwise segmentation** to recover trailed sources that are often missed by traditional algorithms, with a focus on:
- LSST / Rubin Observatory–like data,
- realistic injected datasets,
- reproducible training and evaluation.

---

## Features

- PyTorch U-Net–based architectures with residual and attention blocks
- Robust preprocessing (MAD normalization, sigma clipping)
- HDF5-based tiled datasets for large focal-plane images
- Multi-stage and curriculum training strategies
- Pixel-level and object-level evaluation metrics
- Two-stage detection concepts (connectivity + scoring)
- Compatibility with Rubin Butler–based data products

---

## Typical workflow

1. **Dataset creation**
   - Inject synthetic asteroid trails into single-visit images
   - Store images, masks, and metadata in HDF5 / CSV format

2. **Training**
   - Train segmentation networks on tiled image data
   - Use class-imbalanced losses and staged training

3. **Evaluation**
   - Pixelwise ROC / F1 / AUC
   - Object-level detection via connected components
   - Comparison with LSST stack detections

---

## Requirements (indicative)

- Python ≥ 3.9
- PyTorch
- NumPy, SciPy, pandas
- h5py
- matplotlib
- Rubin Science Pipelines (for data generation and injections)

Exact environments depend on whether you are running on:
- Rubin USDF / SDF
- local workstation
- HPC cluster (SLURM)

---

## Status

Active research and development.  
The codebase evolves alongside:
- Rubin commissioning data (ComCam / LSSTCam)
- improved injection realism
- new detection post-processing strategies
