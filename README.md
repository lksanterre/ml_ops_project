

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

- **data_preprocessing.ipynb**: A Jupyter notebook for preprocessing and cleaning the raw data into a format suitable for model training.

- **model_selection.ipynb**: A Jupyter notebook for experimenting with different models to find the optimal one. It uses the cleaned data to train and evaluate three different models:
  - CNN with GlobalMaxPooling
  - Bidirectional LSTM
  - Simple LSTM

- **mlruns/**: A directory created by MLflow to log and track model training experiments.

- **mlflow.db**: SQLite database used by MLflow for tracking experiment runs.

- **requirements.txt**: Lists all the Python dependencies required to run the notebooks and scripts in this repository.

- **streamlit_app.py**: A Streamlit application for showcasing the question classification model. Users can input questions and see how the model classifies them into 'What,' 'How,' or 'Why.'

## Workflow

1. **Data Preparation**: The data from the Stanford and Quora datasets are preprocessed and cleaned in `data_preprocessing.ipynb`. The cleaned data is then stored as a pickle file in the `data/cleaned` directory.

2. **Weak Supervision for Labeling**: Questions are labeled using a weak supervision technique. The `weak_supervision_label` function categorizes questions based on key phrases and their positions in the question. This function applies different weights depending on the position of the keywords to determine the probability of each question type.

3. **Model Training and Evaluation**: `model_selection.ipynb` is used to train and evaluate three different neural network models: CNN with GlobalMaxPooling, Bidirectional LSTM, and Simple LSTM. The notebook includes a grid search to experiment with different hyperparameters and select the best-performing model. 

4. **Handling Reproducibility**: Efforts were made to address challenges with model reproducibility. The final model and tokenizer were saved in `.h5` and `.pkl` formats, respectively, to ensure consistent predictions.

5. **Showcasing the Model**: A Streamlit application, `streamlit_app.py`, was developed to provide an interactive interface where users can input questions and see the model’s classification results in real-time.

6. **Tracking Experiments**: MLflow is used to track model training experiments, including hyperparameters, performance metrics, and model artifacts.

## Requirements

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## How to Use

1. Clone this repository.
2. Run `data_preprocessing.ipynb` to preprocess the raw data.
3. Use `model_selection.ipynb` to train and evaluate different models.
4. Save the model and tokenizer as `.h5` and `.pkl` files for reproducibility.
5. Run `streamlit_app.py` to interact with the model and see its predictions.
6. Track model performance and tuning experiments using MLflow.

## Goal

The goal of this project is to develop a robust question classification model that can accurately categorize questions into one of three types ("What," "How," or "Why") and produce a 1x3 vector representation for each question. The project also aims to address challenges in model reproducibility and provide an interactive interface for users to engage with the model.

