# Self-Supervised Ensemble Learning for Skin Lesion Classification (HAM10000)

## 🚀 Project Overview

This project implements an advanced computer vision pipeline for multi-class skin lesion classification using the HAM10000 dataset (7 disease categories). The architecture combines:

- Self-Supervised Learning (SimCLR) for representation learning from unlabeled images
- Transfer learning with pretrained backbones (ResNet50, EfficientNetB0)
- Limited-label training scenario (20% labeled data) to emulate real-world annotation constraints
- Ensemble soft-voting of three expert models for robust performance

## 🧠 Key Contributions

- **Limited annotation setting**: trained over just 20% of labels for supervised tasks, demonstrating data-efficient performance.
- **Self-supervised pretraining**: trained a SimCLR encoder on all available training samples (unlabeled) with NT-Xent contrastive loss.
- **Fine-tuning strategy**: used frozen/fine-tuning schedules on pretrained encoders with class-weighted cross-entropy.
- **Model ensemble**: combined ResNet50 (supervised), EfficientNetB0 (supervised), and SimCLR-enhanced ResNet50.
- **Detailed evaluation**: classification report, confusion matrix, per-class ROC curves, and multiclass ROC-AUC.

## 📁 Repository Structure

- `ssl-ensemble-simCLR.ipynb`: complete end-to-end notebook with data preparation, modeling, training, and evaluation.

## 🧩 Dataset: HAM10000 (Skin Cancer MNIST)

- Source: Kaggle (dataset includes dermatoscopic images labeled with pathology class)
- Classes:
  - `akiec` (Actinic Keratoses and Intraepithelial Carcinoma)
  - `bcc` (Basal Cell Carcinoma)
  - `bkl` (Benign Keratosis-like Lesions)
  - `df` (Dermatofibroma)
  - `mel` (Melanoma)
  - `nv` (Melanocytic Nevi)
  - `vasc` (Vascular Lesions)
- Preprocessing:
  - 224x224 image resizing
  - ResNet50 preprocessing pipeline (`preprocess_input`)
  - augmentations: flip, rotation, contrast, brightness, zoom, random resized crop

## ⚙️ Core Implementation Details

### 1. Data Handling

- Stratified `train/val/test` split: 70/10/20
- Limited labeled subset: 20% of training samples selected per class
- Class-weight balancing using `sklearn.utils.class_weight.compute_class_weight`
- `tf.data` pipelines with efficient I/O and caching (`AUTOTUNE`) and augmentations

### 2. Supervised Model Architecture

- ResNet50 and EfficientNetB0 backbones with ImageNet weights
- GlobalAveragePooling + Dropout + Dense softmax head
- Hyperparameters:
  - `BATCH_SIZE=32`, `IMG_SIZE=224x224`
  - `optimizer=Adam(1e-4)`, `loss=sparse_categorical_crossentropy`
  - `unfreeze_last_N` (40/80 layers for fine-tuning)

### 3. SimCLR SSL Pretraining

- Generator creates two augmented views of each input image
- Projection head: Dense(512->128) + L2 normalization
- Contrastive loss: normalized temperature-scaled cross-entropy (NT-Xent, temperature=0.2)
- Pretraining epochs: 30

### 4. SSL-based Classifier

- Reuses pretrained ResNet50 encoder base + classification head
- Optionally freeze earlier layers before fine-tuning

### 5. Ensemble Strategy

- Soft-voting average of predicted probabilities from:
  - `resnet50_model`
  - `effnetb0_model`
  - `ssl_model`
- Predict the class with highest average probability

## 📊 Evaluation and Metrics

- Accuracy and loss curves for train/validation
- Per-model classification reports (precision, recall, F1-score)
- Ensemble classification report
- Ensemble confusion matrix
- Per-class ROC curves + ROC-AUC scores for each class

## 🛠️ Training Setup

In the notebook, training uses:

- `callbacks`: `ReduceLROnPlateau`, `EarlyStopping` (restore best weights)
- 10 epochs of supervised fine-tuning per model (after SSL pretraining)
- class weights in loss to mitigate class imbalance

## 📝 How to Reproduce

1. Install dependencies:
   - `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
2. Place HAM10000 data in the expected path (or adjust `ROOT` variable):
   - `HAM10000_metadata.csv`
   - image folders: `ham10000_images_part_1`, `ham10000_images_part_2`, `HAM10000_images`, `ham10000_images`
3. Run `ssl-ensemble-simCLR.ipynb` from top to bottom

<!-- ## 💎 Resume Bullet Points (Suggested)

- Developed a self-supervised learning pipeline for skin lesion classification using SimCLR + ResNet50, enabling strong representation learning on unlabeled clinical images.
- Applied limited-supervision protocol (20% labels) with class-weighted fine-tuning and advanced augmentation to address severe class imbalance.
- Built an ensemble of ResNet50, EfficientNetB0, and SSL-enhanced ResNet50 with soft probability voting, achieving improved robustness and multiclass generalization.
- Implemented full evaluation suite including confusion matrices, multiclass ROC-AUC, and per-class performance diagnostics. -->

<!-- ## 🗂️ Notes

- This implementation is designed for research and prototyping; full productionization requires:
  - checkpointing, logging, and model versioning,
  - proper dataset pipeline for large-scale training,
  - clinical compliance and rigorous validation.

--- -->

### 📌 Contact

For questions or further improvements, refer to the notebook `ssl-ensemble-simCLR.ipynb` and adapt it to your dataset or deployment environment.
