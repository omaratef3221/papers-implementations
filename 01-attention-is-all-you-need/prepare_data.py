from torch.utils.data import Dataset, DataLoader
from Tokenization.tokenize_data import tokenize_data

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset.map(tokenize_data, batched= True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


