# Breast Cancer Classification using Artificial Neural Network (ANN)

A binary classification model built with an Artificial Neural Network (ANN) to detect Malignant vs Benign breast cancer tumors using the Wisconsin Diagnostic Dataset.

---

## Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for patient survival. This project applies a deep learning approach (ANN) to classify tumors as Malignant (cancerous) or Benign (non-cancerous) based on 30 numeric cell nucleus features.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 95.61% |
| Validation Accuracy | 98.90% |
| Malignant Recall | 96% |
| False Negatives (missed cancers) | Only 3 out of 72 |
| Total Trainable Parameters | 4,097 |
| Training Epochs | 18 |

---

## Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset

| Attribute | Details |
|---|---|
| Total Samples | 569 |
| Features | 30 (all numeric) |
| Classes | Malignant (1), Benign (0) |
| Malignant Samples | 357 (62.7%) |
| Benign Samples | 212 (37.3%) |
| Missing Values | None |
| Source | sklearn.datasets.load_breast_cancer() |

The 30 features represent cell nucleus measurements including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

---

## ANN Architecture

```
Input Layer     ->  30 features
Hidden Layer 1  ->  64 neurons, ReLU activation
Dropout         ->  0.3 rate
Hidden Layer 2  ->  32 neurons, ReLU activation
Dropout         ->  0.3 rate
Output Layer    ->  1 neuron, Sigmoid activation
```

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 30 | — | 0 |
| Hidden 1 + Dropout | 64 | ReLU | 1,984 |
| Hidden 2 + Dropout | 32 | ReLU | 2,080 |
| Output | 1 | Sigmoid | 33 |
| Total | | | 4,097 |

---

## Tech Stack

- Language: Python 3.8+
- Deep Learning: TensorFlow / Keras
- ML & Preprocessing: Scikit-learn
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/YourUsername/breast-cancer-classification-ANN.git
cd breast-cancer-classification-ANN
```

2. Install required libraries
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

3. Run the project
```bash
python breast_cancer_ann.py
```

---

## Training Results (18 Epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 76.92% | 91.21% | 0.5525 | 0.3961 |
| 7 | 96.70% | 97.80% | 0.1151 | 0.0904 |
| 8 | 96.70% | 98.90% | 0.1120 | 0.0809 |
| 18 | 98.35% | 98.90% | 0.0638 | 0.0469 |

- Model converged at Epoch 8 with 98.90% validation accuracy
- No overfitting observed — train/val gap only 0.55%

---

## Confusion Matrix (Test Set — 114 Samples)

```
                  Predicted: Benign   Predicted: Malignant
Actual: Benign         40 (TN)              2 (FP)
Actual: Malignant       3 (FN)             69 (TP)
```

- Only 3 false negatives — missed malignant tumors
- 97% Precision for Malignant class

---

## Sample Prediction

```
Sample Index     : 5
Actual Label     : Malignant (1)
Predicted Prob   : 0.9952  (99.52% confidence)
Predicted Class  : Malignant (1)
Result           : Correct Prediction
```

---

## Key Concepts Used

- StandardScaler — Normalizes features to mean=0, std=1 for stable training
- Dropout (0.3) — Prevents overfitting by randomly disabling neurons
- ReLU Activation — Introduces non-linearity in hidden layers
- Sigmoid Activation — Outputs probability for binary classification
- Binary Crossentropy — Loss function for binary classification
- Adam Optimizer — Adaptive learning rate for fast convergence

---

## Author

Raza Ahmad Khan
- University of Engineering & Technology Mardan
- Department of Computer Science (AI Section)
- Registration No: 23MDBCS452
- Semester: 6th

---

## License

This project is open source and available under the MIT License.
