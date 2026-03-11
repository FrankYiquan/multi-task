## UTKFace Multi-Task Prediction with EfficientNet-B0

This project trains a multi-task deep learning model in PyTorch to predict facial attributes from the UTKFace dataset.
The model jointly predicts **age (regression)**, **gender (binary classification)**, and **race/ethnicity (5-class classification)**.

The final model achieves approximately:

* **Age MAE:** 5.12
* **Gender Accuracy:** 92.45%
* **Race Accuracy:** 82.54%

---

### Project Overview

We train a multi-task convolutional neural network using:

* PyTorch + torchvision
* Pretrained EfficientNet-B0 backbone
* Adam optimizer with weight decay
* Data augmentation
* Label smoothing for classification tasks
* Validation-based model checkpointing

The original EfficientNet classifier is removed and replaced with three task-specific prediction heads for age, gender, and race.

---

Setup

Install required packages:

```
pip install -r requirements.txt
```

---

### Run Training

To train the model:

```
python main.py
```

The best checkpoint will be saved to:

```
output/model.pt. (used this for grading)
```

---

### Run Evaluation

To evaluate the trained model on the test set:

```
python test.py
```

The script reports:

* Age MAE
* Gender Accuracy
* Race Accuracy

---

Project Structure

```
utkface-multitask/
│
├── train.py                # training workflow
├── test.py                 # evaluation script
├── config.py               # hyperparameter configuration
├── main.py                 # training entry point
│
├── model/
│   └── model.py            # MultiTaskEfficientNet architecture
│
├── utils/
│   ├── dataset.py          # UTKFace dataset loader
│   └── eval.py             # evaluation metrics
│
├── data/
│
├── output/
│   └── model.pt            # saved best model(used this for grading)
│
└── requirement.txt         # dependencies needed to be downloaded
```

---

Notes

* Labels are parsed directly from UTKFace image filenames.
* The dataset itself is not included in the repository. The full dataset can be downloaded from:
https://drive.google.com/file/d/1ifjLtLefohfpV-3-HbJeV8nXBgt8X_2Q/view?usp=sharing
After downloading and extracting the dataset, place it inside the data/ directory with the following structure:

```
├── data/
│   ├── train/
│   ├── val/
│   └── test/
```