from dataset import CreditCardDataset, df
from sklearn.model_selection import train_test_split

import torch

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import json

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Target'])

train_dataset = CreditCardDataset(train_df)
val_dataset = CreditCardDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define the model
class CreditCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 12)
        self.fc2 = nn.Linear(12, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # ReLU to introduce non-linearity   
        x = self.fc2(x)     
        return x


model = CreditCardModel()


# Define the loss function
criterion = nn.CrossEntropyLoss()


# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model

for epoch in range(10):
    for i , (inputs, lables) in enumerate(train_loader):
        optimizer.zero_grad()  # to clear the gradients of the previous iteration
        outputs = model(inputs)  # forward pass
        loss = criterion(outputs, lables)  # calculate the loss
        loss.backward()  # backward pass
        optimizer.step()  # update the weights

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Evaluate the model

total_correct = 0
total_samples = 0

with torch.no_grad():
    for i , (inputs, lables) in enumerate(val_loader):
        output = model(inputs)
        predicted = torch.argmax(output, dim=1)
        total_correct += (predicted == lables).sum().item()
        total_samples += lables.size(0)

    print(f"Accuracy: {100 * total_correct / total_samples}")


# Save the model to artifacts folder 
torch.save(model.state_dict(), './artifacts/model.pth')


# config for API 

config = {
    'input_dim': 23,
    'hidden_dim': 12,
    'num_classes': 2,
    'model_path': './artifacts/model.pth',
}

with open('./artifacts/config.json', 'w') as f:
    json.dump(config, f)
