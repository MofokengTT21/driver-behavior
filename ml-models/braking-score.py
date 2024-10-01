import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
import copy
import json

# File paths
file_paths = {
    'smooth': '../data/safe-driving.csv',
    'harsh': '../data/aggressive-driving.csv'
}

# Load and label datasets
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

# Encode categorical label
label_encoder = LabelEncoder()
combined_data['label'] = label_encoder.fit_transform(combined_data['label'])

# Separate features and target
X = combined_data.drop(['label'], axis=1)
y = combined_data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for LSTM (samples, time steps, features)
sequence_length = 1
def create_sequences(data, labels, seq_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        sequence_labels.append(labels.iloc[i + seq_length - 1])  # Use iloc for integer-based indexing
    return np.array(sequences), np.array(sequence_labels)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define improved LSTM model
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h0, c0):
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = X_train_tensor.shape[2]
hidden_size = 64
num_classes = len(label_encoder.classes_)
num_epochs = 20
learning_rate = 0.001

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedLSTMModel(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

# Define early stopping parameters
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
patience = 5
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        h0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
        c0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
        outputs = model(X_batch, h0, c0)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    # Learning rate scheduler step
    scheduler.step()

    # Validation
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            h0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
            c0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
            outputs = model(X_batch, h0, c0)
            _, predicted = torch.max(outputs, 1)
            total_preds += y_batch.size(0)
            correct_preds += (predicted == y_batch).sum().item()

    epoch_acc = correct_preds / total_preds
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset):.4f}, Accuracy: {epoch_acc:.4f}')

    # Check for early stopping
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_model_wts)

# Evaluation and storing correct/incorrect predictions
correct_counts = {label: 0 for label in range(num_classes)}
total_counts = {label: 0 for label in range(num_classes)}

model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        h0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
        c0 = torch.zeros(2, X_batch.size(0), hidden_size).to(device)
        outputs = model(X_batch, h0, c0)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

        for true, pred in zip(y_batch.cpu().numpy(), predicted.cpu().numpy()):
            total_counts[true] += 1
            if true == pred:
                correct_counts[true] += 1

# Print classification report
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Create DataFrame to store the counts
df_counts = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Correct': [correct_counts[i] for i in range(num_classes)],
    'Total': [total_counts[i] for i in range(num_classes)]
})

df_counts['Accuracy'] = df_counts['Correct'] / df_counts['Total']
print(df_counts)

# Save the model in ONNX format
dummy_input = torch.randn(1, sequence_length, input_size).to(device)
h0 = torch.zeros(2, 1, hidden_size).to(device)
c0 = torch.zeros(2, 1, hidden_size).to(device)
torch.onnx.export(model, (dummy_input, h0, c0), "improved_braking_lstm_model.onnx", 
                  input_names=['input', 'h0', 'c0'], output_names=['output'])
print("Model saved to improved_braking_lstm_model.onnx")

# Save scaler mean and variance
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'var': scaler.var_.tolist()
}

with open('./scaler/braking_scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)
