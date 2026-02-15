# Read the dataset file from data/data.xls and then wrap the dataset in torch.utils.data.Dataset 
import pandas as pd
from torch.utils.data import Dataset
import torch

df = pd.read_excel('./data/data.xls', header=1)
df.rename(columns={'default payment next month': 'Target'}, inplace=True)
df = df.drop(columns=['ID'])

class CreditCardDataset(Dataset):    # subclassing the Dataset class from torch.utils.data
    def __init__(self, df):
        self.df = df
        self.X = df.drop(columns=['Target'])
        self.y = df['Target']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
        return x, y



if __name__ == "__main__":
    dataset = CreditCardDataset(df)
    print(len(dataset))
    print(dataset[0])

    print(dataset[22])

