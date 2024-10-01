import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load new dataset
new_data_file_path = '../data/aggressive-driving.csv'
new_data = pd.read_csv(new_data_file_path)

# Load and prepare the training data
file_paths = {
    'safe': '../data/safe-driving.csv',
    'rushed': '../data/rushed-driving.csv',
    'aggressive': '../data/aggressive-driving.csv'
}

def load_and_label(file_path, label):
    df = pd.read_csv(file_path)
    df['label'] = label
    return df

data_frames = []
for label, file_path in file_paths.items():
    df = load_and_label(file_path, label)
    data_frames.append(df)

combined_data = pd.concat(data_frames, ignore_index=True)

X_train = combined_data.drop(['label'], axis=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Standardize new data
new_data_scaled = scaler.transform(new_data)

# Define sequence length
sequence_length = 1
num_samples = new_data_scaled.shape[0]

# Load ONNX model
onnx_model_path = 'improved_lstm_model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Print input names and shapes
for input_meta in session.get_inputs():
    print(f"Input name: {input_meta.name}, Shape: {input_meta.shape}")

# Prepare input data
input_name = session.get_inputs()[0].name  # Name of the input layer in ONNX model
h0_name = session.get_inputs()[1].name     # Name of the h0 input layer
c0_name = session.get_inputs()[2].name     # Name of the c0 input layer

print(f"Input name: {input_name}")
print(f"h0 name: {h0_name}")
print(f"c0 name: {c0_name}")

# Initialize hidden and cell states for LSTM
hidden_size = 64
batch_size = 1  # Set batch size to 1

# Prepare to collect predictions
predictions = []

# Define label mapping for your model's output
label_mapping = {
    0: "aggressive",
    1: "rushed",
    2: "safe",
}

# Process each sequence
for start in range(num_samples - sequence_length + 1):
    end = start + sequence_length
    sequence = new_data_scaled[start:end]  # Shape: [10, num_features]

    # Reshape to match expected input
    sequence = np.reshape(sequence, (1, sequence_length, sequence.shape[1]))  # Shape: [1, 10, num_features]
    sequence = sequence.astype(np.float32)

    # Initialize hidden and cell states for each sequence
    h0 = np.zeros((2, batch_size, hidden_size), dtype=np.float32)
    c0 = np.zeros((2, batch_size, hidden_size), dtype=np.float32)

    input_data = {
        input_name: sequence,
        h0_name: h0,
        c0_name: c0
    }

    # Make predictions
    try:
        prediction = session.run(None, input_data)
        # Check the shape of the prediction output
        print(f"Prediction shape: {prediction[0].shape}")

        if prediction[0].ndim == 3:
            predicted_classes = np.argmax(prediction[0], axis=2)  # Axis 2 for sequence length
            predictions.extend(predicted_classes.flatten())
        elif prediction[0].ndim == 2:
            predicted_classes = np.argmax(prediction[0], axis=1)  # Axis 1 for class probabilities
            predictions.extend(predicted_classes)

    except Exception as e:
        print(f"An error occurred while processing sequence starting at index {start}: {e}")

# Convert indices to labels
predicted_labels = [label_mapping[idx] for idx in predictions]

# Print the predicted labels
print("Predicted driving statuses for the new dataset:")
print(predicted_labels)

# (Optional) Save the predictions to a CSV file
predicted_df = pd.DataFrame({'Predicted_Driving_Status': predicted_labels})
predicted_df.to_csv('../data/predicted_new_data.csv', index=False)

print("Predictions saved to predicted_new_data.csv")
