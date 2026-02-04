# AMS 325 Final Project: Handwritten Digit Classification

## 📌 Project Overview
This project presents a comparative study of classical machine learning methods (**Random Forest, Boosted Trees**) and deep learning models (**Neural Networks**) on the **MNIST** handwritten digit dataset. 

The primary objective was to evaluate and analyze the trade-offs between **model complexity**, **computational cost**, and **classification accuracy** across different architectures.

* **Course:** AMS 325 (Stony Brook University)
* **Team Members:** Brandon Park, Jungjae Park, Kyunghwan Min, Minjae Lee, Minseo Jang

---

## 💡 Key Design Decisions & Terminology
During the model architecture design phase, our team established clear definitions to distinguish between "Shallow" and "Deep" learning within the context of this project:

* **Defining the "Single Layer" Network:** * While a Perceptron maps inputs directly to outputs (0 hidden layers), we determined that to meaningfully evaluate a neural network's capability against tree-based models, the baseline network must possess non-linear feature extraction capabilities.
    * Therefore, in this repository, **"Single Layer NN"** is defined as a network with **exactly one hidden layer** (Input $\rightarrow$ Hidden $\rightarrow$ Output), serving as our baseline for neural architectures.
* **Defining the "Multi-Layer" Network:** * To test the effect of depth, **"Multi-Layer NN"** refers to a network with **two or more hidden layers**, designed to observe hierarchical feature learning.

---

## 🛠 Models Implemented

### 1. Random Forest (`sklearn`)
* **Objective:** To establish a strong classical baseline resistant to overfitting.
* **Methodology:** Conducted hyperparameter tuning on `mtry` (number of features considered at each split).
* **Finding:** Using $\sqrt{p}$ features outperformed $0.5p$, confirming that decorrelating trees improves ensemble performance.

### 2. Boosted Trees (`XGBoost`)
* **Objective:** To push classification accuracy by sequentially correcting errors.
* **Methodology:** * **Binary Task (R):** Isolated digits 3 vs. 8 to analyze overfitting behavior in depth.
    * **Multiclass Task (Python):** applied to the full 0-9 dataset.
* **Finding:** Achieved the highest classification accuracy among all models but incurred the highest computational cost.

### 3. Neural Networks (`TensorFlow/Keras`)
* **Single Hidden Layer NN:** * **Architecture:** `Input (784) -> Dense(150, ReLU) -> Dropout(0.2) -> Output(10, Softmax)`
    * **Role:** Acts as a bridge between simple linear models and deep learning.
* **Multi-Hidden Layer NN:**
    * **Architecture:** `Input -> Dense(150) -> Dropout -> Dense(50) -> Dropout -> Output`
    * **Finding:** Demonstrated that distributing model capacity across multiple layers allows for more efficient bias reduction compared to a single wide layer.

---

## 📂 Repository Structure

* `FINAL_Project.py`: The main Python script integrating Random Forest, Multiclass XGBoost, and Neural Network implementations.
* `Boosted_Tree.R`: R script dedicated to the Binary Classification (3 vs 8) experiment for granular performance analysis.
* `AMS325_Project Report.pdf`: Full project report containing theoretical background and detailed error analysis.
* `AMS325_Project Presentation.pptx`: Presentation slides summarizing the project workflow and results.

---

## 📊 Performance Summary

| Model | Configuration | Key Result |
| :--- | :--- | :--- |
| **Random Forest** | `mtry = sqrt(p)` | **Robust Baseline:** Fast training with high stability. |
| **XGBoost** | Depth 5, 20k rounds | **Best Accuracy:** Superior performance but computationally intensive. |
| **Single Layer NN** | 150 Neurons | **Competitive:** Good feature extraction, but limited capacity. |
| **Multi Layer NN** | 150 $\rightarrow$ 50 Neurons | **Efficiency:** Improved accuracy over Single Layer via hierarchical structure. |

---

## 🚀 Usage

### Dependencies
* Python 3.x
* Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`

### Execution
To reproduce the Python results (RF, XGBoost, NN):
```bash
python FINAL_Project.py
