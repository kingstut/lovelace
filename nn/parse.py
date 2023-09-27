import json
import os
import torch
from sklearn.model_selection import train_test_split

def load_data(relay_dir, tensor_ir_dir):
    relay_files = os.listdir(relay_dir)
    tensor_ir_files = os.listdir(tensor_ir_dir)

    relay_data = []
    tensor_ir_data = []

    for fname in relay_files:
        with open(os.path.join(relay_dir, fname), 'r') as f:
            relay_data.append(json.load(f))

    for fname in tensor_ir_files:
        with open(os.path.join(tensor_ir_dir, fname), 'r') as f:
            tensor_ir_data.append(json.load(f))

    return relay_data, tensor_ir_data

def preprocess_data(relay_data, tensor_ir_data):
    # Convert the structured JSON data into a numerical format suitable for a neural network
    # This is a placeholder and would involve tokenization, encoding, etc.
    # For simplicity, let's assume a dummy preprocessing step
    X = [len(r) for r in relay_data]  # Example: number of nodes in Relay IR
    y = [len(t) for t in tensor_ir_data]  # Example: number of nodes in Tensor IR

    return X, y

relay_data, tensor_ir_data = load_data('relay_ir_data/', 'tensor_ir_data/')
X, y = preprocess_data(relay_data, tensor_ir_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
