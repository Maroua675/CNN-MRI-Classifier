# Brain Tumor MRI Classification with a CNN

A convolutional neural network (TensorFlow/Keras) that classifies brain MRI images into four classes: **glioma, meningioma, pituitary tumor, and no tumor**.

> ⚠️ **Disclaimer:** This is an educational project. It is not a medical device and must not be used for diagnosis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/CNN_MRI_Classifier.ipynb)

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle).

| Split | Images per class | Total |
|-------|------------------|-------|
| Training | 1,400 | 5,600 |
| Testing | 400 | 1,600 |

The classes are balanced. Images come in different sizes (mostly 512x512) and color modes, so they are resized to 224x224 RGB. The training folder is split 80/20 into training and validation sets, and the test set is kept separate until final evaluation.

The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub`.

## Project workflow

1. Environment setup (GPU on Google Colab)
2. Dataset download and structure check
3. Exploratory data analysis: class balance, image sizes, color modes, pixel values
4. Preprocessing: resizing to 224x224, pixel normalization (0-1), data augmentation (rotation, zoom, translation, contrast)
5. CNN model building, training, and evaluation
6. Evaluation: accuracy, confusion matrix, precision/recall/F1 per class

## Model

A CNN built from scratch:

- Augmentation and `Rescaling(1/255)` layers inside the model
- 4 convolutional blocks (Conv2D + BatchNormalization + MaxPooling)
- `GlobalAveragePooling2D`, Dense(256), Dropout(0.5)
- Softmax output with 4 classes
- Adam optimizer, sparse categorical cross-entropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Results

| Version | Description | Test accuracy | Glioma recall |
|---------|-------------|---------------|---------------|
| v1 | Baseline CNN, no augmentation or normalization | 85.6% | 0.76 |
| v2 | Improved CNN with augmentation, normalization, BatchNorm, callbacks | _TBD_ | _TBD_ |

Full per-class metrics and the confusion matrix are in the notebook.

### Observations

- The v1 baseline overfit: training accuracy reached about 95% while test accuracy stayed near 86%.
- Recall on glioma and meningioma was the weakest, which matters most in a medical setting because missed tumors are the costly error.

## How to run

1. Open the notebook in Google Colab (use the badge above).
2. Set the runtime to GPU: Runtime → Change runtime type → T4 GPU.
3. Run all cells. The dataset downloads automatically.

To run locally:

```bash
pip install -r requirements.txt
jupyter notebook CNN_MRI_Classifier.ipynb
```

## Requirements

```
tensorflow
kagglehub
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
```

## Future improvements

- Transfer learning (MobileNetV2, ResNet50, EfficientNet) and comparison with the scratch CNN
- Grad-CAM visualizations to see which regions drive predictions
- Error analysis of misclassified images
- Patient-level splitting to avoid slice leakage between training and validation

## Acknowledgements

Dataset: Masoud Nickparvar, Brain Tumor MRI Dataset, Kaggle.
