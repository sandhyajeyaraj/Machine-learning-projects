# Deep Learning Programming - COVID-19 CT Classification

This project focuses on building a deep learning model to classify **COVID-19** and **Non-COVID-19** cases from CT scan images.

## Dataset
The dataset contains:
- **1000+ COVID-19 CT images** from 216 patients
- **1000+ Non-COVID-19 CT images**

The dataset is organized into **train** and **test** folders.

## Task
The goal is to develop an expert model using deep learning techniques to effectively classify COVID and Non-COVID CT scans.

## Requirements
- Reduce model overfitting using appropriate techniques (e.g., dropout, data augmentation, regularization)
- Explore model performance for different learning rates and output layer regularization
- Apply callbacks to:
  - Save the best model
  - Stop training early if validation performance does not improve for `n` consecutive epochs (early stopping)

---