# Read 
import json

import torch.nn as nn
import torch

with open('./artifacts/config.json', 'r') as f:
    config = json.load(f)

input_dim = config['input_dim']
hidden_dim = config['hidden_dim']
num_classes = config['num_classes']
model_path = config['model_path']

# Define the model
class CreditCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # ReLU to introduce non-linearity   
        x = self.fc2(x)     
        return x

def load_model():
    model = CreditCardModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model