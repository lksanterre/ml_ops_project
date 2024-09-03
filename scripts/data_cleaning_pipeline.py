import numpy as np
import pandas as pd
import pickle
import os

# Function for applying weak supervision labeling
def weak_supervision_label(question):
    if not isinstance(question, str):
        return [0.33, 0.33, 0.33]  # Default uncertain label for non-string values

    question = question.lower()
    
    what_like = ["what", "which", "who", "where", "when"]
    how_like = ["how", "in what way", "by what means"]
    why_like = ["why", "for what reason", "how come"]

    label = np.array([0.0, 0.0, 0.0])
    weight_start = 1.5
    weight_middle = 1.0
    weight_end = 2.0

    def apply_weights(word_list, weight):
        nonlocal label
        for word in word_list:
            if word in question:
                if question.endswith(word):
                    label += np.array([weight_end if word in what_like else 0, 
                                        weight_end if word in how_like else 0, 
                                        weight_end if word in why_like else 0])
                elif question.startswith(word):
                    label += np.array([weight_start if word in what_like else 0, 
                                       weight_start if word in how_like else 0, 
                                       weight_start if word in why_like else 0])
                else: 
                    label += np.array([weight_middle if word in what_like else 0, 
                                       weight_middle if word in how_like else 0, 
                                       weight_middle if word in why_like else 0])

    apply_weights(what_like, weight_middle)
    apply_weights(how_like, weight_middle)
    apply_weights(why_like, weight_middle)

    if np.sum(label) > 0:
        label /= np.sum(label)
    else:
        label = np.array([0.33, 0.33, 0.33])  # Uncertain label
    
    return label.tolist()

# Load data from CSV and JSON
data_csv = pd.read_csv('/Users/lancesanterre/intern_2024/data/uncleaned/q_quora.csv', low_memory=False)
data_json = pd.read_json('/Users/lancesanterre/intern_2024/data/uncleaned/dev-v1.1.json')

# Extract questions from JSON
questions = []
for entry in data_json["data"]:
    for paragraph in entry["paragraphs"]:
        for qa in paragraph["qas"]:
            questions.append(qa["question"])

# Add questions from CSV
questions.extend(data_csv['question1'].dropna().tolist())

# Create DataFrame
df = pd.DataFrame(questions, columns=['question'])

# Apply labeling
df['labels'] = df['question'].apply(weak_supervision_label)

# Save the DataFrame and pipeline in a single pickle file
output_file = '/Users/lancesanterre/intern_2024/data/processed/pipeline_and_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(df, f)

print("Pipeline and data saved successfully.")
