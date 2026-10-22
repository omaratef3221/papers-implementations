from torch.utils.data import Dataset, DataLoader
from Tokenization.tokenize_data import tokenize_data

import datasets

def get_dataset(variation = 'de-en', split = True):
    if split:
        split_amt = {"train": "train[:1%]","validation": "validation[:1%]","test": "test[:1%]"}
    dataset = datasets.load_dataset('wmt/wmt14', variation,
                            split= split_amt
                                )
    return dataset
    
    

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset.map(tokenize_data, batched= True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]