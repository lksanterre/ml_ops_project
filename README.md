

# intern_education

## Overview

This repository contains code and data for training a model to categorize questions into one of three types: "What," "How," and "Why." The aim is to output a 1x3 vector for each question, where:
- The first element represents a "What" question,
- The second element represents a "How" question, and
- The third element represents a "Why" question.

## Project Structure

- **data/**: Contains both raw and cleaned datasets.
  - *Uncleaned Data*: Question data from the Stanford Question Answering Dataset (SQuAD) and Quora.
  - *Cleaned Data*: Processed data stored as a pickle file.

- **model_selection.ipynb**: A Jupyter notebook for experimenting with different models to find the optimal one. It uses the cleaned data to train and evaluate three different models:
  - CNN with GlobalMaxPooling
  - Bidirectional LSTM
  - Simple LSTM

- **first_model.ipynb**: A Jupyter notebook for preprocessing and cleaning the raw data into a format suitable for model training.

- **mlruns/**: A directory created by MLflow to log and track model training experiments.

- **mlflow.db**: SQLite database used by MLflow for tracking experiment runs.

- **requirements.txt**: Lists all the Python dependencies required to run the notebooks and scripts in this repository.

## Workflow

1. **Data Preparation**: The data from the Stanford and Quora datasets are preprocessed and cleaned in `first_model.ipynb`. The cleaned data is then stored as a pickle file in the `data/cleaned` directory.

2. **Model Training and Evaluation**: `model_selection.ipynb` is used to train and evaluate three different neural network models: CNN with GlobalMaxPooling, Bidirectional LSTM, and Simple LSTM. The notebook is designed to experiment with different hyperparameters to determine the best performing model.

3. **Tracking Experiments**: MLflow is used to track model training experiments, including hyperparameters, performance metrics, and model artifacts.

## Requirements

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## How to Use

1. Clone this repository.
2. Run `first_model.ipynb` to preprocess the raw data.
3. Use `model_selection.ipynb` to train and evaluate different models.
4. Track the model performance and tuning experiments using MLflow.

## Goal

The goal of this project is to develop a robust question classification model that can accurately categorize questions into one of three types ("What," "How," or "Why") and produce a 1x3 vector representation for each question.
