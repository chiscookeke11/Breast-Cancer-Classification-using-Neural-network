# Breast Cancer Classification Using Neural Network

A machine learning project that uses a **Neural Network** to classify breast cancer tumors as **Malignant** or **Benign** using diagnostic features from the Breast Cancer Wisconsin dataset.

The project demonstrates an end-to-end machine learning workflow, including data loading, preprocessing, feature encoding, standardization, neural network construction, model training, evaluation, and making predictions on new data.

> **Disclaimer:** This project is intended for educational and research purposes only. It is **not a medical diagnostic system** and should not be used to make clinical decisions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Objective](#-project-objective)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Data Preprocessing](#-data-preprocessing)
- [Neural Network Architecture](#-neural-network-architecture)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Making Predictions](#-making-predictions)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Understanding the Prediction](#-understanding-the-prediction)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

## 🔎 Overview

Breast cancer classification is a binary classification problem where a machine learning model learns to distinguish between two classes:

- **Malignant (M)** — represented as `0`
- **Benign (B)** — represented as `1`

This project uses a feed-forward neural network built with **TensorFlow/Keras** to learn patterns from numerical measurements associated with breast cell nuclei.

The model receives **30 numerical features** for each observation and produces a prediction for one of the two classes.

---

## 🎯 Project Objective

The primary objective of this project is to build and evaluate a neural network capable of classifying breast cancer observations based on diagnostic measurements.

The project also demonstrates how to:

- Load and inspect a CSV dataset
- Clean unnecessary columns
- Encode categorical target values
- Separate features from the target
- Split data into training and testing sets
- Standardize numerical features
- Build a neural network with Keras
- Train a neural network
- Monitor training and validation performance
- Evaluate the model on unseen test data
- Use the trained model to make predictions

---

## ❓ Problem Statement

Given a collection of numerical measurements describing characteristics of breast cell nuclei, can a neural network learn to determine whether an observation represents a **malignant** or **benign** tumor?

This can be formulated as a supervised binary classification problem:

```text
Input:
30 numerical diagnostic features

        ↓

Neural Network

        ↓

Output:
Malignant (0)
or
Benign (1)
```

---

## 📊 Dataset

The project uses a breast cancer dataset stored locally in:

```text
sample_data/data.csv
```

The dataset contains:

- **569 observations**
- **33 columns** before preprocessing
- A `diagnosis` column containing the target
- Numerical measurements describing characteristics of cell nuclei

The original dataset contains two diagnosis classes:

| Diagnosis | Meaning | Encoded Value |
|-----------|---------|---------------|
| M | Malignant | `0` |
| B | Benign | `1` |

### Class Distribution

After encoding the target:

| Class | Samples |
|-------|---------:|
| Benign | 357 |
| Malignant | 212 |
| **Total** | **569** |

The dataset therefore contains more benign observations than malignant observations, making it important to consider class distribution when evaluating the model.

---

## 🔄 Machine Learning Workflow

The project follows this general pipeline:

```text
                    ┌─────────────────┐
                    │     Dataset     │
                    │    data.csv     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature / Target│
                    │   Separation    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Target Encoding │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Train/Test Split│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ StandardScaler  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Neural Network  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Training  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Evaluation│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Prediction   │
                    └─────────────────┘
```

---

## 🛠 Technologies Used

The project was implemented using:

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| Jupyter Notebook | Interactive development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Data splitting and feature standardization |
| TensorFlow / Keras | Neural network development and training |

---

## 📁 Project Structure

```text
Breast-Cancer-Classification-using-Neural-network/
│
├── NN_breast_cancer_classification.ipynb
│
├── sample_data/
│   └── data.csv
│
└── README.md
```

### `NN_breast_cancer_classification.ipynb`

The main Jupyter Notebook containing the complete machine learning workflow.

It includes:

- Dependency imports
- Dataset loading
- Data exploration
- Data cleaning
- Feature encoding
- Train/test splitting
- Feature standardization
- Neural network creation
- Model training
- Accuracy/loss visualization
- Model evaluation
- Prediction

### `sample_data/data.csv`

The dataset used to train and evaluate the neural network.

---

# 🧹 Data Preprocessing

Before training the model, several preprocessing operations are performed.

## 1. Loading the Dataset

The dataset is loaded using Pandas:

```python
breast_cancer_dataset = pd.read_csv("./sample_data/data.csv")
```

---

## 2. Removing Unnecessary Columns

The original dataset contains:

- `id`
- `Unnamed: 32`

These columns are removed because they are not useful as predictive features.

```python
breast_cancer_dataset.drop(
    columns=["Unnamed: 32", "id"],
    axis=1,
    inplace=True
)
```

After removing these columns, the dataset contains the diagnostic features and target column.

---

## 3. Encoding the Target

The `diagnosis` column originally contains:

```text
M
B
```

These values are converted into numerical labels:

```python
M → 0
B → 1
```

The encoding is performed using:

```python
breast_cancer_dataset["diagnosis"] = (
    breast_cancer_dataset["diagnosis"]
    .replace({
        "M": 0,
        "B": 1
    })
)
```

This allows the neural network to work with numerical target values.

---

## 4. Separating Features and Target

The dataset is separated into:

### Features — `X`

The input variables used by the neural network.

```python
X = breast_cancer_dataset.drop(
    columns="diagnosis",
    axis=1
)
```

### Target — `Y`

The diagnosis that the model is expected to predict.

```python
Y = breast_cancer_dataset["diagnosis"]
```

The model ultimately works with **30 input features**.

---

## 5. Train/Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

This means:

- **80%** of the data is used for training
- **20%** is reserved for testing

With 569 total observations, the test set contains **114 observations**.

---

## 6. Feature Standardization

The numerical features have very different scales.

For example, some measurements may be fractions while others may represent values in the hundreds or thousands.

To make training more stable, `StandardScaler` is used.

```python
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

X_train_std = Scaler.fit_transform(X_train)
X_test_std = Scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform both training and testing data.

This is important because the test set should not influence the fitting of the preprocessing transformation.

---

# 🧠 Neural Network Architecture

The neural network is built using Keras' `Sequential` API.

The current architecture is:

```text
Input
30 features
   │
   ▼
Flatten
   │
   ▼
Dense Layer
20 neurons
ReLU activation
   │
   ▼
Dense Layer
2 neurons
Sigmoid activation
   │
   ▼
Output
2 classes
```

The implementation is:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid"),
])
```

### Layer Breakdown

#### Flatten Layer

```python
keras.layers.Flatten(input_shape=(30,))
```

Receives the 30 input features and ensures the input is represented in a format suitable for the dense layers.

#### Hidden Layer

```python
keras.layers.Dense(20, activation="relu")
```

This layer contains:

- 20 neurons
- ReLU activation

ReLU introduces non-linearity, allowing the network to learn more complex relationships between the input features.

#### Output Layer

```python
keras.layers.Dense(2, activation="sigmoid")
```

The output layer contains two neurons corresponding to the two possible classes.

---

# ⚙️ Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer

```text
Adam
```

Adam is used to update the model's weights during training.

### Loss Function

```text
Sparse Categorical Crossentropy
```

The loss function measures the difference between the model's predictions and the actual class labels.

### Metric

```text
Accuracy
```

Accuracy measures the percentage of predictions that match the actual labels.

---

# 🏋️ Model Training

The model is trained for **10 epochs**.

A validation split of **10% of the training data** is also used:

```python
history = model.fit(
    X_train_std,
    Y_train,
    validation_split=0.1,
    epochs=10
)
```

This allows the model's performance to be monitored on data that is not directly used for updating its weights.

---

# 📈 Training Results

The model's performance improved significantly throughout training.

| Epoch | Training Accuracy | Validation Accuracy |
|------:|-------------------:|---------------------:|
| 1 | 49.63% | 73.91% |
| 2 | 71.15% | 84.78% |
| 3 | 81.17% | 91.30% |
| 4 | 85.82% | 95.65% |
| 5 | 89.00% | 95.65% |
| 6 | 89.98% | 95.65% |
| 7 | 91.69% | 95.65% |
| 8 | 93.40% | 95.65% |
| 9 | 94.13% | 95.65% |
| 10 | **95.11%** | **95.65%** |

The training accuracy increased from approximately **49.63% to 95.11%**, while validation accuracy reached approximately **95.65%**.

The notebook also visualizes:

- Training vs. validation accuracy
- Training vs. validation loss

These plots help identify how the model learns over time.

---

# 🧪 Model Evaluation

After training, the model is evaluated against the held-out test set:

```python
loss, accuracy = model.evaluate(
    X_test_std,
    Y_test
)
```

The recorded test performance is:

```text
Test Loss:     0.1540
Test Accuracy: 0.9474
```

### Final Test Accuracy

**94.74%**

This means the model correctly classified approximately 94.74% of the observations in the test set.

With 114 test observations, this corresponds to approximately:

```text
108 correctly classified
6 incorrectly classified
```

> Accuracy alone does not provide a complete picture of a medical classification model's performance. Metrics such as precision, recall, specificity, sensitivity, F1-score, ROC-AUC, and a confusion matrix would provide a more complete evaluation.

---

# 🔮 Making Predictions

After training, the model can be used to classify a new observation.

An example input containing 30 diagnostic features is supplied:

```python
input_data = (
    20.57, 17.77, 132.9, 1326, 0.08474,
    0.07864, 0.0869, 0.07017, 0.1812,
    0.05667, 0.5435, 0.7339, 3.398, 74.08,
    0.005225, 0.01308, 0.0186, 0.0134,
    0.01389, 0.003532, 24.99, 23.41,
    158.8, 1956, 0.1238, 0.1866,
    0.2416, 0.186, 0.275, 0.08902
)
```

The input is converted into a NumPy array:

```python
input_data_as_np_array = np.asarray(input_data)
```

It is then reshaped:

```python
input_data_reshaped = input_data_as_np_array.reshape(1, -1)
```

The same scaler used during training is applied:

```python
input_data_std = Scaler.transform(input_data_reshaped)
```

Finally, the model generates a prediction:

```python
prediction = model.predict(input_data_std)
```

The predicted class is selected using:

```python
prediction_label = [np.argmax(prediction)]
```

---

# 🧩 Understanding the Prediction

The model produces two output values representing the model's scores for the two classes.

For example:

```text
[0.6109636, 0.18063736]
```

The class with the larger value is selected using `argmax`.

The project's encoding is:

```text
0 → Malignant
1 → Benign
```

Therefore:

```python
np.argmax(prediction)
```

returns the class with the highest predicted score.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/chiscookeke11/Breast-Cancer-Classification-using-Neural-network.git
```

Navigate into the project:

```bash
cd Breast-Cancer-Classification-using-Neural-network
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

Alternatively, if you are using Jupyter directly, you can install TensorFlow with:

```python
%pip install tensorflow
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
NN_breast_cancer_classification.ipynb
```

Then execute the notebook cells from top to bottom.

Make sure the dataset remains at:

```text
sample_data/data.csv
```

because the notebook loads it using:

```python
pd.read_csv("./sample_data/data.csv")
```

---

# 📚 What You Can Learn From This Project

This project is useful for understanding several important machine learning concepts.

### Data Science

- Loading datasets
- Exploring tabular data
- Cleaning data
- Feature selection
- Target encoding

### Machine Learning

- Supervised learning
- Classification
- Training/testing split
- Feature scaling
- Model evaluation
- Prediction

### Neural Networks

- Input layers
- Dense layers
- Activation functions
- ReLU
- Sigmoid
- Loss functions
- Optimizers
- Epochs
- Validation data

### TensorFlow/Keras

- `Sequential`
- `Dense`
- `Flatten`
- `model.compile()`
- `model.fit()`
- `model.evaluate()`
- `model.predict()`

---

# ⚠️ Limitations

Although the model achieves approximately **94.74% test accuracy**, several limitations should be considered.

## 1. Small Dataset

The model is trained on only 569 observations.

A larger and more diverse dataset would be required before considering real-world deployment.

## 2. Accuracy Is Not Enough

For medical classification, simply reporting accuracy can be misleading.

A future version should report:

- Precision
- Recall
- Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix

## 3. No External Validation

The model is evaluated using a held-out portion of the same dataset.

Testing against an independent external dataset would provide stronger evidence of generalization.

## 4. Neural Network Architecture

The current architecture is intentionally simple:

```text
30 → 20 → 2
```

More systematic experimentation could determine whether a different architecture performs better.

## 5. Medical Use

The model should **not** be used as a substitute for professional medical diagnosis.

A real clinical system would require extensive validation, appropriate clinical datasets, regulatory considerations, interpretability, monitoring, and expert oversight.

---

# 🚀 Future Improvements

Several improvements could make the project more robust.

### 1. Add More Evaluation Metrics

Implement:

```text
Confusion Matrix
Precision
Recall
F1 Score
ROC-AUC
Specificity
Sensitivity
```

### 2. Improve the Neural Network

Experiment with:

- More hidden layers
- Different numbers of neurons
- Dropout
- Batch normalization
- Different activation functions

For example:

```text
30
 ↓
64 neurons
 ↓
32 neurons
 ↓
16 neurons
 ↓
2 classes
```

### 3. Use Early Stopping

Instead of always training for a fixed number of epochs:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

could be used to stop training when validation performance stops improving.

### 4. Hyperparameter Tuning

Experiment with:

- Learning rate
- Batch size
- Number of layers
- Number of neurons
- Optimizers
- Activation functions
- Number of epochs

### 5. Improve Prediction Output

Instead of returning only:

```text
0
```

or:

```text
1
```

the prediction system could return:

```text
Prediction: Benign
Confidence: 92.4%
```

### 6. Build an Interactive Interface

The trained model could be connected to an application using:

- Streamlit
- FastAPI
- Flask
- React/Next.js

This would allow users to enter feature values through a user interface and receive a prediction.

### 7. Improve Model Architecture

The current output layer uses two neurons with sigmoid activation. For a two-class classification problem, a future implementation could also experiment with a single output neuron using sigmoid activation and binary cross-entropy, or a two-neuron softmax output with sparse categorical cross-entropy.

---

# 🔬 Possible Production Architecture

A more complete version of this project could look like:

```text
                User Input
                    │
                    ▼
             Input Validation
                    │
                    ▼
            Feature Preprocessing
                    │
                    ▼
              Trained Model
                    │
                    ▼
             Prediction Engine
                    │
                    ▼
          ┌────────────────────┐
          │ Classification     │
          │                    │
          │ Malignant / Benign │
          └────────────────────┘
                    │
                    ▼
             Result + Metrics
```

A production system would also require appropriate security, privacy, monitoring, validation, and clinical governance.

---

# 📖 Project Learning Journey

This project represents a practical introduction to building a neural-network-based classification system from a structured dataset.

The workflow moves from:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Data Standardization
   ↓
Neural Network
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
```

This makes the project useful as a learning reference for beginners studying **Machine Learning, Artificial Intelligence, and Neural Networks**.

---

# 👨‍💻 Author

**Chinedu Okeke Emmanuel**

Software Engineer | AI/ML Engineer | Open Source Contributor

GitHub: `@chiscookeke11`

---

# ⭐ If You Find This Useful

If this project helps you understand neural networks or machine learning classification, consider giving the repository a star and exploring the notebook.

---

## 📜 License

No license has currently been specified for this repository.

If you intend for others to freely use, modify, and distribute the project, consider adding an appropriate open-source license such as the MIT License.
