import torch
import torch.nn as nn
import torch.onnx

import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input", help="input size")
argParser.add_argument("-h", "--hidden", help="hidden size")
argParser.add_argument("-o", "--output", help="output size")

# Define a simple two-layered MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def main():
    args = argParser.parse_args()

    # Instantiate the model
    input_size = args.i or 10
    h_size = args.h or 5
    output_size = args.o or 1
    model = SimpleMLP(input_size, h_size, output_size)
    model.eval()

    # Dummy input for ONNX export
    x = torch.randn(1, input_size)

    # Export the model to ONNX
    torch.onnx.export(model, x, f"../onnx_saved/mlp_{input_size}_{h_size}_{output_size}.onnx")
