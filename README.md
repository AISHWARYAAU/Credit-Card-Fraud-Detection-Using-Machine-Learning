# Credit Card Fraud Detection using Autoencoder and Logistic Regression

This project demonstrates a machine learning approach to detect credit card fraud using an autoencoder for feature extraction and logistic regression for classification. 

## Introduction

Credit card fraud detection is a critical task for financial institutions to protect customers and prevent monetary losses. Traditional methods often rely on rule-based systems or simple anomaly detection techniques. In this project, we explore a more sophisticated approach using deep learning and logistic regression.

## Data

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with only 492 fraud transactions out of 284,807 transactions.

## Approach

1. **Exploratory Data Analysis (EDA)**: We perform basic data exploration to understand the distribution of transactions and the amount involved in fraud and non-fraud transactions.

2. **Data Preprocessing**: We preprocess the data by scaling the "Amount" column using Min-Max scaling and splitting it into training and testing sets.

3. **Autoencoder Model**: We train an autoencoder neural network to learn the underlying patterns in the data. The autoencoder reconstructs the input data, and the middle layer's output serves as a compressed representation of the input.

4. **Feature Extraction**: We extract features from the compressed representations obtained from the autoencoder.

5. **Logistic Regression Model**: We train a logistic regression classifier using the extracted features to classify transactions as fraudulent or non-fraudulent.

6. **Evaluation**: We evaluate the performance of the model using classification metrics such as accuracy, precision, recall, and F1-score.

## Usage

1. **Data Preparation**: Download the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

2. **Running the Code**: Execute the provided Python script `credit_card_fraud_detection.py` in your Python environment.

## Repository Structure

```
.
├── credit_card_fraud_detection.py    # Python script for credit card fraud detection
├── creditcard.csv                    # Credit card fraud dataset
└── README.md                         # This README file
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tensorflow
- scikit-learn
- keras

