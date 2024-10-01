import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load new dataset for braking score predictions
new_data_file_path = '../data/aggressive-driving.csv'  # Path to new braking data
new_data = pd.read_csv(new_data_file_path)

# File paths for the training data (braking)
file_paths = {
    'smooth': '../data/safe-driving.csv',
    'harsh': '../data/aggressive-driving.csv'
}

# Function to load and label datasets
def load_and_label(file_path, label):
    df = pd.read_csv(file_path)
    df['label'] = label
    return df

# Combine datasets
data_frames = []
for label, file_path in file_paths.items():
    df = load_and_label(file_path, label)
    data_frames.append(df)

combined_data = pd.concat(data_frames, ignore_index=True)

# Encode labels and separate features and target
combined_data['label'] = combined_data['label'].astype('category').cat.codes
X_train = combined_data.drop(['label'], axis=1)
y_train = combined_data['label']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Define sequence length
sequence_length = 1
num_samples = new_data_scaled.shape[0]

# Load the ONNX model
onnx_model_path = 'improved_braking_lstm_model.onnx'  # Path to the ONNX braking model
session = ort.InferenceSession(onnx_model_path)

# Get input names and shapes
for input_meta in session.get_inputs():
    print(f"Input name: {input_meta.name}, Shape: {input_meta.shape}")

# Get input and hidden state names
input_name = session.get_inputs()[0].name
h0_name = session.get_inputs()[1].name
c0_name = session.get_inputs()[2].name

# Initialize LSTM hidden and cell states
hidden_size = 64  # Make sure this matches the hidden size in your model
batch_size = 1  # Batch size is 1 for each sequence

# Prepare to collect predictions
predictions = []

# Define label mapping (modify based on your braking score classes)
label_mapping = {
    0: 'harsh',
    1: 'smooth'
}

# Iterate over sequences in the new data
for start in range(num_samples - sequence_length + 1):
    end = start + sequence_length
    sequence = new_data_scaled[start:end]

    # Reshape the sequence to match the input shape expected by the model
    sequence = np.reshape(sequence, (1, sequence_length, sequence.shape[1])).astype(np.float32)

    # Initialize hidden and cell states
    h0 = np.zeros((2, batch_size, hidden_size), dtype=np.float32)
    c0 = np.zeros((2, batch_size, hidden_size), dtype=np.float32)

    input_data = {
        input_name: sequence,
        h0_name: h0,
        c0_name: c0
    }

    # Run inference
    try:
        prediction = session.run(None, input_data)
        print(f"Prediction shape: {prediction[0].shape}")

        # Process prediction output
        if prediction[0].ndim == 3:  # For sequence outputs
            predicted_classes = np.argmax(prediction[0], axis=2)  # Axis 2 for classes
            predictions.extend(predicted_classes.flatten())
        elif prediction[0].ndim == 2:  # For batch outputs
            predicted_classes = np.argmax(prediction[0], axis=1)  # Axis 1 for class probabilities
            predictions.extend(predicted_classes)

    except Exception as e:
        print(f"Error while processing sequence starting at index {start}: {e}")

# Convert numeric predictions to labels
predicted_labels = [label_mapping[idx] for idx in predictions]

# Print the predicted braking statuses
print("Predicted braking statuses for the new dataset:")
print(predicted_labels)

print("Predictions saved to predicted_braking_data.csv")
