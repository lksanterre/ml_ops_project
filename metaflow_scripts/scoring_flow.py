# training_flow.py
from metaflow import FlowSpec, step, Parameter
import mlflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from data_processing import process_data  # Import your data processing module

class TrainingFlow(FlowSpec):

    # Parameters for the flow
    data_path = Parameter('data_path', default='/Users/lancesanterre/intern_2024/data/cleaned/data.pkl', help="Path to the new data for training")
    input_dim = Parameter('input_dim', default=1000, help="Vocabulary size for the tokenizer")
    output_dim = Parameter('output_dim', default=64, help="Output dimension for the Embedding layer")
    input_length = Parameter('input_length', default=10, help="Input length for padding sequences")

    @step
    def start(self):
        """Start step: Load and preprocess the new data."""
        print("Starting the training flow...")
        self.data = process_data(self.data_path)
        self.next(self.tokenize_and_transform)

    @step
    def tokenize_and_transform(self):
        """Tokenize and transform the data."""
        # Extract questions and labels from the processed data
        questions = self.data[0]  # Adjust key based on your data structure
        labels = self.data[1]  # Adjust key based on your data structure
        
        # Tokenization and Padding
        self.tokenizer = Tokenizer(num_words=self.input_dim)
        self.tokenizer.fit_on_texts(questions)
        sequences = self.tokenizer.texts_to_sequences(questions)
        self.X = pad_sequences(sequences, maxlen=self.input_length)

        # Convert labels to numpy array
        self.y = np.array(labels.tolist())  # Adjust if needed
        
        # Save the tokenizer as a .pkl file
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print("Tokenizer saved as 'tokenizer.pkl'")
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model on the entire dataset."""
        print("Training the model...")
        # Define the model architecture
        self.model = Sequential([
            Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length),
            LSTM(self.output_dim),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model on the entire dataset
        self.model.fit(self.X, self.y, epochs=10, verbose=2)  # Adjust epochs as needed
        self.next(self.save_predictions)

    @step
    def save_predictions(self):
        """Feed the model input to the trained model, predict, and save the predictions."""
        # Use the trained model to make predictions on the input data
        predictions = self.model.predict(self.X)
        
        # Save the predictions using pickle
        prediction_path = '/Users/lancesanterre/intern_2024/data/predictions/predictions.pkl'
        with open(prediction_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        print(f"Predictions saved as '{prediction_path}'")
        self.next(self.end)
    @step
    def end(self):
        """End step."""
        print("Training flow completed!")

if __name__ == '__main__':
    TrainingFlow()
