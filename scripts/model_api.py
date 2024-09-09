import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the tokenizer from the pickle file
with open('/Users/lancesanterre/intern_2024/notebooks/Bidirectional_LSTM_1000_64_2_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the Keras model from the HDF5 file
model = load_model('/Users/lancesanterre/intern_2024/notebooks/Bidirectional_LSTM_1000_64_2.h5')

# Define the input dimension and settings
input_dim = 1000
input_length = 10  # Ensure this matches the maxlen used during training

# Initialize session state if not already done
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# Streamlit app
st.title("Question Classification Model")
st.write("Enter a question to get the classification vector.")

# User input
user_question = st.text_input("Enter your question:", value=st.session_state.user_question)

if st.button("Predict"):
    # Tokenize and pad new data using the loaded tokenizer
    sequences = tokenizer.texts_to_sequences([user_question])
    X_new = pad_sequences(sequences, maxlen=input_length)  # Use the same maxlen as used during training

    # Make predictions
    predictions = model.predict(X_new)
    predictions_percent = np.round(predictions * 100, 2)
    predictions_hundredth = np.round(predictions, 2)

    # Update session state
    st.session_state.prediction = {
        'percent': predictions_percent,
        'hundredth': predictions_hundredth
    }
    st.session_state.user_question = user_question

if st.session_state.prediction:
    # Display results
    st.write("Predictions as percentages:", st.session_state.prediction['percent'])
    st.write("Predictions rounded to the hundredth place:", st.session_state.prediction['hundredth'])

# Reset button to clear results
if st.button("Reset"):
    st.session_state.prediction = None
    st.session_state.user_question = ""
    st.experimental_rerun()  # Rerun the app to clear inputs and predictions
