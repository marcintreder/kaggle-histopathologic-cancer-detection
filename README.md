# Histopathologic Cancer Detection using Convolutional Neural Networks

This repository contains the code and analysis for the "Histopathologic Cancer Detection" mini-project, part of the Deep Learning coursework at the University of Colorado Boulder. The goal is to build and train a CNN to identify metastatic cancer in small image patches of lymph node sections.

The project follows a systematic, iterative approach to model development, starting with a simple baseline and progressively improving the architecture based on performance analysis.

---

## 📂 Project Structure

-   `histopathologic_cancer_detection.ipynb`: The main Jupyter Notebook containing all the code, analysis, and visualizations.
-   `histopathologic_cancer_detection.pdf`: PDF with the output of the Jupyter Notebook (for easier review of the analysis)
-   `submission.csv`: The final prediction file for the Kaggle competition leaderboard.
-   `histopathologic-cancer-detection/`: The directory containing the training and test image data (not included in this repo due to size).

---

## 🔬 Methodology

The project was structured as a series of experiments to find the optimal model architecture and hyperparameters.

### 1. Exploratory Data Analysis (EDA)

The initial analysis confirmed that the dataset was large (~220,000 images), well-balanced between the two classes (cancer vs. no-cancer), and required no significant data cleaning. Visual inspection revealed that the patterns distinguishing the classes were very subtle, justifying the need for a powerful deep learning model.

### 2. Data Preparation

A memory-efficient data pipeline was built using `tf.data.Dataset`. This approach was chosen over loading the entire dataset into memory to prevent RAM-related crashes. The pipeline reads, decodes, and normalizes the `.tif` image files in small batches, allowing for efficient training even on a local machine.

### 3. Iterative Model Development

A series of models were built and trained to compare architectures and tune hyperparameters:

* **Model v1 (Baseline):** A minimal CNN with one convolutional block. **Result:** Learned fast but was highly unstable and overfit almost immediately, demonstrating that a more robust architecture was needed.
* **Model v2 (Champion Model):** A deeper, regularized architecture with two convolutional blocks, `BatchNormalization`, and `Dropout`. **Result:** This model trained in a stable, healthy manner, achieving a **peak validation AUC of ~0.87**.
* **Model v3 (Too Deep):** An experiment with a three-block CNN. **Result:** The model was too deep to train effectively from scratch and proved to be highly unstable.
* **Model v4 (Too Wide):** An experiment with a wider two-block CNN. **Result:** This model also proved to be unstable, confirming that `model_v2` represented a "sweet spot" of complexity.
* **Final Tuning:** The champion `model_v2` was re-tested with an `RMSprop` optimizer, which was found to be more erratic than the original `Adam` optimizer.

---

## 📊 Results

The final, best-performing model was **`model_v2`**, which used a two-block architecture, `BatchNormalization`, `Dropout`, the `Adam` optimizer, and a learning rate of `0.0001`. This combination provided the best balance of model capacity and training stability.

| Model | Peak `val_auc` | Analysis & Findings |
| :--- | :---: | :--- |
| **Model v1** | ~0.89 | Unstable, overfit immediately. |
| **Model v2** | **~0.87** | **Stable & Successful.** The champion model. |
| **Model v3** | ~0.84 | Failed (Too Deep). |
| **Model v4** | ~0.82 | Failed (Too Wide). |



---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    ```
2.  **Set up the environment:** Ensure you have Python 3.9+ and install the required libraries:
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn opencv-python jupyterlab
    ```
3.  **Download the data:** Download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/histopathologic-cancer-detection/data) and unzip it into the main project directory.
4.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
5.  Open and run the `Cancer_Detection_Project.ipynb` notebook.
